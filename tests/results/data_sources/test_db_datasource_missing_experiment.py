import pytest
from experiment_manager.results.sources.db_datasource import DBDataSource

# Removed unused import of mnist_db_datasource_fixture


def test_get_nonexistent_experiment_by_int_id(experiment_data):
    """Querying a non-existent experiment by integer ID should raise ValueError with a clear message."""
    db_path = experiment_data['db_path']
    with DBDataSource(db_path) as mnist_db:
        nonexistent_id = 999999  # Use a very large id to ensure it does not exist
        with pytest.raises(ValueError) as excinfo:
            mnist_db.get_experiment(nonexistent_id)
        msg = str(excinfo.value).lower()
        assert "does not exist" in msg or "not found" in msg


def test_get_nonexistent_experiment_by_str_id(experiment_data):
    """Querying a non-existent experiment by string ID should raise ValueError with a clear message."""
    db_path = experiment_data['db_path']
    with DBDataSource(db_path) as mnist_db:
        with pytest.raises(ValueError) as excinfo:
            mnist_db.get_experiment("this_experiment_does_not_exist")
        msg = str(excinfo.value).lower()
        assert "does not exist" in msg or "not found" in msg


def test_get_nonexistent_trial_for_valid_experiment(experiment_data):
    """Querying a non-existent trial for a valid experiment should raise ValueError or return None/empty, depending on API."""
    db_path = experiment_data['db_path']
    with DBDataSource(db_path) as mnist_db:
        experiment = mnist_db.get_experiment()
        nonexistent_trial_id = 999999
        if hasattr(mnist_db, "get_trial"):
            with pytest.raises(ValueError) as excinfo:
                mnist_db.get_trial(experiment.id, nonexistent_trial_id)
            msg = str(excinfo.value).lower()
            assert "does not exist" in msg or "not found" in msg
        elif hasattr(mnist_db, "get_trials"):
            trials = mnist_db.get_trials(experiment)
            assert all(trial.id != nonexistent_trial_id for trial in trials)
        else:
            pytest.skip("DBDataSource does not have get_trial or get_trials method") 