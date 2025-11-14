"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import set_seed, Timer, load_config


class TestSetSeed:
    """Test random seed setting."""
    
    def test_set_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_seed(42)
        result1 = np.random.rand(10)
        
        set_seed(42)
        result2 = np.random.rand(10)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        result1 = np.random.rand(10)
        
        set_seed(123)
        result2 = np.random.rand(10)
        
        assert not np.array_equal(result1, result2)


class TestTimer:
    """Test Timer context manager."""
    
    def test_timer_measures_time(self):
        """Test that Timer correctly measures elapsed time."""
        import time
        
        with Timer("Test") as timer:
            time.sleep(0.1)
        
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Should be close to 0.1
    
    def test_timer_elapsed_property(self):
        """Test elapsed property."""
        import time
        
        timer = Timer("Test")
        assert timer.elapsed == 0.0
        
        with timer:
            time.sleep(0.05)
        
        assert timer.elapsed > 0


@pytest.mark.skip(reason="Config file may not exist in test environment")
class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_config(self):
        """Test loading configuration file."""
        config = load_config("conf/params.yaml")
        assert isinstance(config, dict)
        assert 'seed' in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
