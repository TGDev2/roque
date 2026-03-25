"""Tests for the play mode controller."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import chess

from chesslab.analysis import AnalysisConfig
from chesslab.analysis import ContinuousUpdate, Evaluation
from chesslab.capture.screen_capture import CapturePermissionError
from chesslab.coach.game_coach import ContinuousAdvice, PlayerColor
from chesslab.gui.controllers import AppMode, ReviewModeRequest
from chesslab.gui.controllers.play import PlayController
from chesslab.gui.live_observer import LiveObserverConfig
from chesslab.validation import CalibrationMetrics, CalibrationReport, load_support_catalog

from tests.gui.controller_test_utils import DummyHost


def _make_report(*, passed: bool) -> CalibrationReport:
    return CalibrationReport(
        setup_id="chesscom_green_white",
        support_level="official",
        template_theme="chess.com-green",
        expected_theme="chess.com/green",
        passed=passed,
        blockers=()
        if passed
        else ("Chessboard detection is too unstable (50.0% < 90.0%).",),
        warnings=() if passed else ("Setup support level is 'candidate', not 'official'.",),
        corrective_actions=()
        if passed
        else (
            "Keep the full chessboard visible on the selected monitor and confirm "
            "you chose the monitor that actually contains the board.",
        ),
        metrics=CalibrationMetrics(
            frame_count=4,
            detection_rate=1.0 if passed else 0.5,
            theme_match_rate=1.0,
            ready_rate=1.0 if passed else 0.25,
            uncertain_rate=0.0,
            invalid_rate=0.0,
            p95_total_latency_ms=120.0,
            p95_detection_latency_ms=20.0,
            p95_split_latency_ms=10.0,
            p95_recognition_latency_ms=10.0,
            status_counts={"ready": 4} if passed else {"board_not_found": 2, "ready": 2},
        ),
    )


def _make_checkmate_board() -> chess.Board:
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    return board


class TestObserveLiveHelpers:
    def test_format_preflight_report_surfaces_actions(self) -> None:
        profile = load_support_catalog().get_profile("chesscom_green_white")
        assert profile is not None

        text = PlayController._format_preflight_report(profile, _make_report(passed=False))

        assert "Blockers:" in text
        assert "Corrective actions:" in text
        assert "Chessboard detection is too unstable" in text
        assert "Keep the full chessboard visible" in text

    def test_format_preflight_report_marks_pass_as_technical_only(self) -> None:
        profile = load_support_catalog().get_profile("chesscom_green_white")
        assert profile is not None

        text = PlayController._format_preflight_report(profile, _make_report(passed=True))

        assert "technical compatibility only" in text
        assert "outside tools" in text

    def test_format_preflight_error_uses_actionable_capture_message(self) -> None:
        message = PlayController._format_preflight_error(
            CapturePermissionError("permission denied")
        )

        assert "permission denied" in message.lower()
        assert "screen recording" in message.lower()


class TestPlayController:
    def _make_controller(self) -> tuple[DummyHost, PlayController]:
        host = DummyHost()
        controller = PlayController(host, initial_player_color=PlayerColor.WHITE)
        return host, controller

    def test_start_observation_reports_preflight_pass_for_declared_setup(self) -> None:
        host, controller = self._make_controller()
        controller._observation_queue = queue.Queue()

        config = LiveObserverConfig(
            monitor_index=1,
            theme="chess.com-green",
            setup_id="chesscom_green_white",
            support_level="official",
        )
        observer = MagicMock()

        with patch("chesslab.gui.controllers.play.LiveObserver", return_value=observer):
            assert controller._start_observation(config) is True

        assert controller._last_observer_config == config
        assert controller._observing is True
        assert observer.start.called
        assert host.set_status.call_args_list[-1].args[0].startswith("Preflight PASS")
        assert "Passive sync only" in host.set_status.call_args_list[-1].args[0]

    def test_start_observation_switches_ui_to_passive_mode(self) -> None:
        host, controller = self._make_controller()
        controller._coach = MagicMock()
        controller._coach.move_history = ("e4",)
        controller._observe_btn = MagicMock()
        controller._apply_btn = MagicMock()
        controller._undo_btn = MagicMock()
        controller._eval_panel = MagicMock()
        controller._observe_panel = MagicMock()
        controller._suggestions_panel = MagicMock()
        controller._history_panel = MagicMock()
        controller._btn_frame = MagicMock()
        controller._new_game_btn = MagicMock()
        controller._pgn_frame = MagicMock()
        controller._entry_frame = MagicMock()
        controller._move_entry = MagicMock()
        controller._submit_btn = MagicMock()
        controller._observation_queue = queue.Queue()

        config = LiveObserverConfig(
            monitor_index=1,
            theme="chess.com-green",
            setup_id="chesscom_green_white",
            support_level="official",
        )
        observer = MagicMock()

        with patch("chesslab.gui.controllers.play.LiveObserver", return_value=observer):
            assert controller._start_observation(config) is True

        controller._coach.stop_analysis.assert_called_once()
        host.widgets.board_widget.set_interactive.assert_called_with(False)
        controller._move_entry.config.assert_any_call(state="disabled")
        controller._submit_btn.config.assert_called_with(state="disabled")
        controller._eval_panel.update_eval.assert_called_with(None, 0, is_loading=False)
        controller._suggestions_panel.update_suggestions.assert_called_with(())
        controller._observe_panel.update_session.assert_called_with(
            setup_label="chesscom_green_white",
            move_count=1,
        )

    def test_submit_san_can_record_passively_without_analysis(self) -> None:
        host, controller = self._make_controller()
        controller._coach = MagicMock()
        controller._clear_coaching_outputs = MagicMock()
        controller._refresh_after_move = MagicMock()

        controller._submit_san("e4", passive=True)

        controller._coach.record_move.assert_called_once_with("e4")
        controller._coach.submit_opponent_move_continuous.assert_not_called()
        controller._clear_coaching_outputs.assert_called_once()
        controller._refresh_after_move.assert_called_once()

    def test_exit_observe_to_review_stops_live_before_review(self) -> None:
        host, controller = self._make_controller()
        controller._observing = True
        controller._coach = MagicMock()
        board = chess.Board()
        board.push_san("e4")
        controller._coach.board = board
        controller._stop_observation = MagicMock()

        controller._exit_observe_to_review()

        controller._stop_observation.assert_called_once()
        mode, request = host.last_switch
        assert mode is AppMode.REVIEW
        assert isinstance(request, ReviewModeRequest)
        assert request.from_observation is True

    def test_apply_advice_shows_manual_coach_hint(self) -> None:
        host, controller = self._make_controller()
        controller._eval_panel = MagicMock()
        controller._suggestions_panel = MagicMock()

        advice = ContinuousAdvice(
            opponent_move_san="e4",
            current_fen=chess.Board().fen(),
            update=ContinuousUpdate(
                fen=chess.Board().fen(),
                variations=(),
                evaluation=Evaluation(centipawns=35, mate_in=None),
                best_move_uci="e2e4",
                best_move_san="e4",
                depth_reached=14,
                elapsed_ms=120.0,
            ),
            move_number=1,
        )

        controller._apply_advice(advice)

        assert controller._best_move_uci == "e2e4"
        host.widgets.board_widget.set_move_hint.assert_called_once_with(
            "e2",
            "e4",
            color="#3b82f6",
        )

    def test_apply_advice_clears_hint_when_no_best_move_is_available(self) -> None:
        host, controller = self._make_controller()
        controller._eval_panel = MagicMock()
        controller._suggestions_panel = MagicMock()
        controller._best_move_uci = "e2e4"

        advice = ContinuousAdvice(
            opponent_move_san="e4",
            current_fen=chess.Board().fen(),
            update=ContinuousUpdate(
                fen=chess.Board().fen(),
                variations=(),
                evaluation=Evaluation(centipawns=0, mate_in=None),
                best_move_uci="",
                best_move_san="",
                depth_reached=14,
                elapsed_ms=120.0,
            ),
            move_number=1,
        )

        controller._apply_advice(advice)

        assert controller._best_move_uci is None
        host.widgets.board_widget.clear_highlights.assert_called_once()

    def test_handle_finished_manual_game_auto_saves_and_offers_review(self) -> None:
        _host, controller = self._make_controller()
        controller._coach = MagicMock()
        controller._coach.board = _make_checkmate_board()
        controller._coach.game_result = "0-1"
        controller._coach.game_termination = None
        controller._clear_coaching_outputs = MagicMock()
        controller._auto_save = MagicMock()
        controller._propose_manual_review = MagicMock()

        controller._handle_finished_manual_game()

        controller._clear_coaching_outputs.assert_called_once()
        controller._auto_save.assert_called_once()
        proposed_board = controller._propose_manual_review.call_args.args[0]
        assert controller._propose_manual_review.call_args.kwargs["result_text"] == "0-1"
        assert proposed_board.result() == "0-1"

    @patch("chesslab.gui.controllers.play.messagebox.askyesno", return_value=True)
    def test_resign_records_selected_side_and_finishes_game(
        self,
        _askyesno: MagicMock,
    ) -> None:
        _host, controller = self._make_controller()
        controller._coach = MagicMock()
        controller._clear_coaching_outputs = MagicMock()
        controller._update_turn_status = MagicMock()
        controller._handle_finished_manual_game = MagicMock()

        controller._on_resign(PlayerColor.WHITE)

        controller._coach.record_resignation.assert_called_once_with(PlayerColor.WHITE)
        controller._clear_coaching_outputs.assert_called_once()
        controller._update_turn_status.assert_called_once()
        controller._handle_finished_manual_game.assert_called_once()

    def test_build_headers_include_manual_resignation_termination(self) -> None:
        host, controller = self._make_controller()
        host.engine_elo = 1500
        controller._coach = MagicMock()
        controller._coach._config.player_color = PlayerColor.WHITE
        controller._coach.game_result = "0-1"
        controller._coach.game_termination = "Black won by resignation"

        headers = controller._build_headers()

        assert headers["Result"] == "0-1"
        assert headers["Termination"] == "Black won by resignation"

    def test_update_turn_status_mentions_resignation_winner(self) -> None:
        host, controller = self._make_controller()
        controller._coach = MagicMock()
        controller._coach.is_game_over = True
        controller._coach.game_winner = chess.BLACK
        controller._coach.game_termination = "Black won by resignation"
        controller._auto_save = MagicMock()

        controller._update_turn_status()

        controller._auto_save.assert_called_once()
        host.set_status.assert_called_once_with(
            "Game over — Black wins by resignation!",
            game_over=True,
        )

    def test_new_game_closes_post_game_review_prompt(self) -> None:
        host, controller = self._make_controller()
        host.show_color_dialog.return_value = (
            PlayerColor.WHITE,
            AnalysisConfig.DEFAULT_ENGINE_ELO,
        )
        controller._auto_save = MagicMock()
        controller._coach = MagicMock()
        controller._eval_panel = MagicMock()
        controller._suggestions_panel = MagicMock()
        controller._history_panel = MagicMock()
        controller._update_turn_status = MagicMock()

        new_coach = MagicMock()
        new_coach.current_fen = chess.Board().fen()
        host.create_coach.return_value = new_coach

        dialog = MagicMock()
        dialog.winfo_exists.return_value = True
        controller._post_game_review_dialog = dialog

        controller._on_new_game()

        dialog.destroy.assert_called_once()
