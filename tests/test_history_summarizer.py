"""Tests for History Summarizer."""

import json
import tempfile
import collections
from pathlib import Path

import pytest

from tamagawa_to_z.tuning.history_summarizer import HistorySummarizer


class TestHistorySummarizer:
    """Test cases for HistorySummarizer."""

    def test_update_and_context(self, tmp_path):
        """Test updating history and generating context."""
        log_file = tmp_path / "test_history.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Add a trial
        weights = {"rio": 0.9, "lago": 0.5}
        fp_roots = collections.Counter({"lagoa": 4, "rio": 2})
        
        summarizer.update(
            distance_km=3.0,
            occ_pct=5.0,
            score=0.3,
            weights=weights,
            fp_roots=fp_roots
        )
        
        # Check that trial was added
        assert len(summarizer.trials) == 1
        assert summarizer.trials[0]["D"] == 3.0
        assert summarizer.trials[0]["O"] == 5.0
        assert summarizer.trials[0]["score"] == 0.3
        assert summarizer.trials[0]["weights"] == weights
        
        # Test context generation
        context = summarizer.to_context()
        assert "best_trials" in context
        assert "history_hash" in context
        assert len(context["best_trials"]) == 1
        assert context["best_trials"][0]["D"] == 3.0
        
        # Check that file was written
        assert log_file.exists()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) == 1
        trial_data = json.loads(lines[0])
        assert trial_data["D"] == 3.0

    def test_multiple_trials_context(self, tmp_path):
        """Test context generation with multiple trials."""
        log_file = tmp_path / "test_history.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Add multiple trials with different scores
        trials_data = [
            (2.0, 3.0, 0.2, {"rio": 0.7}),
            (3.0, 5.0, 0.5, {"lago": 0.8}),
            (4.0, 7.0, 0.3, {"igarape": 0.6})
        ]
        
        for dist, occ, score, weights in trials_data:
            summarizer.update(
                distance_km=dist,
                occ_pct=occ,
                score=score,
                weights=weights,
                fp_roots=collections.Counter()
            )
        
        context = summarizer.to_context(top_m=2)
        
        # Should have 2 best trials (sorted by score)
        assert len(context["best_trials"]) == 2
        
        # First should be highest score (0.5)
        assert context["best_trials"][0]["score"] == 0.5
        assert context["best_trials"][0]["D"] == 3.0
        
        # Second should be next highest (0.3)
        assert context["best_trials"][1]["score"] == 0.3
        assert context["best_trials"][1]["D"] == 4.0

    def test_empty_history_context(self, tmp_path):
        """Test context generation with empty history."""
        log_file = tmp_path / "empty_history.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        context = summarizer.to_context()
        assert context == {}

    def test_fp_roots_accumulation(self, tmp_path):
        """Test false positive roots accumulation."""
        log_file = tmp_path / "test_fp.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Add trials with different FP roots
        summarizer.update(
            distance_km=1.0, occ_pct=2.0, score=0.1,
            weights={"rio": 0.5},
            fp_roots=collections.Counter({"rio": 3, "lago": 1})
        )
        
        summarizer.update(
            distance_km=2.0, occ_pct=3.0, score=0.2,
            weights={"lago": 0.6},
            fp_roots=collections.Counter({"rio": 2, "igarape": 4})
        )
        
        context = summarizer.to_context()
        
        # Check cumulative FP roots ranking
        fp_rank = context["cumulative_fp_roots"]
        assert "rio" in fp_rank  # Should appear first (total: 5)
        assert "igarape" in fp_rank  # Second (total: 4)
        assert "lago" in fp_rank  # Third (total: 1)

    def test_history_hash_changes(self, tmp_path):
        """Test that history hash changes when trials are added."""
        log_file = tmp_path / "test_hash.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Empty history
        context1 = summarizer.to_context()
        hash1 = context1.get("history_hash", "empty")
        
        # Add a trial
        summarizer.update(
            distance_km=1.0, occ_pct=2.0, score=0.1,
            weights={"rio": 0.5},
            fp_roots=collections.Counter()
        )
        
        context2 = summarizer.to_context()
        hash2 = context2["history_hash"]
        
        # Hash should change
        assert hash1 != hash2

    def test_load_existing_history(self, tmp_path):
        """Test loading existing history from file."""
        log_file = tmp_path / "existing.jsonl"
        
        # Create a history file manually
        trial_data = {
            "id": 1,
            "D": 1.5,
            "O": 4.0,
            "score": 0.4,
            "weights": {"rio": 0.8},
            "fp_roots": {"lago": 2}
        }
        
        with log_file.open("w") as f:
            f.write(json.dumps(trial_data) + "\n")
        
        # Load summarizer
        summarizer = HistorySummarizer(log_file)
        
        # Should have loaded the existing trial
        assert len(summarizer.trials) == 1
        assert summarizer.trials[0]["D"] == 1.5
        assert summarizer.trials[0]["score"] == 0.4

    def test_stats_generation(self, tmp_path):
        """Test statistics generation."""
        log_file = tmp_path / "test_stats.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Empty stats
        stats = summarizer.get_stats()
        assert stats["n_trials"] == 0
        
        # Add trials
        for i in range(3):
            summarizer.update(
                distance_km=float(i + 1),
                occ_pct=float(i * 2 + 1),
                score=0.1 * (i + 1),
                weights={f"root_{i}": 0.5},
                fp_roots=collections.Counter()
            )
        
        stats = summarizer.get_stats()
        assert stats["n_trials"] == 3
        assert abs(stats["best_score"] - 0.3) < 1e-10  # Handle floating point precision
        assert stats["distance_range"] == [1.0, 3.0]
        assert stats["occ_range"] == [1.0, 5.0]
        assert stats["unique_roots"] == 3

    def test_clear_history(self, tmp_path):
        """Test clearing history."""
        log_file = tmp_path / "test_clear.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Add a trial
        summarizer.update(
            distance_km=1.0, occ_pct=2.0, score=0.1,
            weights={"rio": 0.5},
            fp_roots=collections.Counter()
        )
        
        assert len(summarizer.trials) == 1
        assert log_file.exists()
        
        # Clear
        summarizer.clear()
        
        assert len(summarizer.trials) == 0
        assert not log_file.exists()

    def test_context_size_limit(self, tmp_path):
        """Test that context stays under size limit."""
        log_file = tmp_path / "test_size.jsonl"
        summarizer = HistorySummarizer(log_file)
        
        # Add many trials
        for i in range(20):
            weights = {f"root_{j}": 0.5 for j in range(10)}  # Many roots
            summarizer.update(
                distance_km=float(i),
                occ_pct=float(i),
                score=0.1 * i,
                weights=weights,
                fp_roots=collections.Counter({f"fp_{j}": j for j in range(5)})
            )
        
        context = summarizer.to_context()
        context_size = len(json.dumps(context, ensure_ascii=False))
        
        # Should be compact (under 2KB as suggested in spec)
        assert context_size < 2000, f"Context size {context_size} exceeds 2KB limit"


def test_create_sample_history():
    """Test the sample history creation function."""
    from tamagawa_to_z.tuning.history_summarizer import create_sample_history
    
    summarizer = create_sample_history()
    
    # Should have created some trials
    assert len(summarizer.trials) > 0
    
    # Should have valid structure
    context = summarizer.to_context()
    assert "best_trials" in context
    assert "history_hash" in context
    
    # Cleanup
    summarizer.clear()


if __name__ == "__main__":
    pytest.main([__file__])