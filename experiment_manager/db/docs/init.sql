-- Create EXPERIMENT table
CREATE TABLE EXPERIMENT (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    `desc` TEXT,
    start_time DATETIME NOT NULL,
    update_time DATETIME NOT NULL
);

-- Create TRIAL table
CREATE TABLE TRIAL (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    experiment_id INT NOT NULL,
    start_time DATETIME NOT NULL,
    update_time DATETIME NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES EXPERIMENT(id)
);

-- Create TRIAL_RUN table
CREATE TABLE TRIAL_RUN (
    id INT PRIMARY KEY AUTO_INCREMENT,
    trial_id INT NOT NULL,
    status VARCHAR(50) NOT NULL,
    start_time DATETIME NOT NULL,
    update_time DATETIME NOT NULL,
    FOREIGN KEY (trial_id) REFERENCES TRIAL(id)
);

-- Create RESULTS table
CREATE TABLE RESULTS (
    trial_run_id INT PRIMARY KEY,
    time DATETIME NOT NULL,
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
);

-- Create EPOCH table
CREATE TABLE EPOCH (
    idx INT,
    trial_run_id INT,
    time DATETIME NOT NULL,
    PRIMARY KEY (idx, trial_run_id),
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
);

-- Create BATCH table
CREATE TABLE BATCH (
    idx INT,
    epoch_idx INT,
    trial_run_id INT,
    time DATETIME NOT NULL,
    PRIMARY KEY (idx, epoch_idx, trial_run_id),
    FOREIGN KEY (epoch_idx, trial_run_id) REFERENCES EPOCH(idx, trial_run_id)
);

-- Create METRIC table
CREATE TABLE METRIC (
    id INT PRIMARY KEY AUTO_INCREMENT,
    type VARCHAR(50) NOT NULL,
    total_val FLOAT NOT NULL,
    per_label_val JSON
);

-- Create ARTIFACT table
CREATE TABLE ARTIFACT (
    id INT PRIMARY KEY AUTO_INCREMENT,
    type VARCHAR(50) NOT NULL,
    loc VARCHAR(255) NOT NULL
);

-- Create relationship tables

-- EXPERIMENT to ARTIFACT relationship
CREATE TABLE EXPERIMENT_ARTIFACT (
    experiment_id INT,
    artifact_id INT,
    PRIMARY KEY (experiment_id, artifact_id),
    FOREIGN KEY (experiment_id) REFERENCES EXPERIMENT(id),
    FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
);

-- TRIAL to ARTIFACT relationship
CREATE TABLE TRIAL_ARTIFACT (
    trial_id INT,
    artifact_id INT,
    PRIMARY KEY (trial_id, artifact_id),
    FOREIGN KEY (trial_id) REFERENCES TRIAL(id),
    FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
);

-- RESULTS to METRIC relationship
CREATE TABLE RESULTS_METRIC (
    results_id INT,
    metric_id INT,
    PRIMARY KEY (results_id, metric_id),
    FOREIGN KEY (results_id) REFERENCES RESULTS(trial_run_id),
    FOREIGN KEY (metric_id) REFERENCES METRIC(id)
);

-- RESULTS to ARTIFACT relationship
CREATE TABLE RESULTS_ARTIFACT (
    results_id INT,
    artifact_id INT,
    PRIMARY KEY (results_id, artifact_id),
    FOREIGN KEY (results_id) REFERENCES RESULTS(trial_run_id),
    FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
);

-- EPOCH to METRIC relationship
CREATE TABLE EPOCH_METRIC (
    epoch_idx INT,
    epoch_trial_run_id INT,
    metric_id INT,
    PRIMARY KEY (epoch_idx, epoch_trial_run_id, metric_id),
    FOREIGN KEY (epoch_idx, epoch_trial_run_id) REFERENCES EPOCH(idx, trial_run_id),
    FOREIGN KEY (metric_id) REFERENCES METRIC(id)
);

-- EPOCH to ARTIFACT relationship
CREATE TABLE EPOCH_ARTIFACT (
    epoch_idx INT,
    epoch_trial_run_id INT,
    artifact_id INT,
    PRIMARY KEY (epoch_idx, epoch_trial_run_id, artifact_id),
    FOREIGN KEY (epoch_idx, epoch_trial_run_id) REFERENCES EPOCH(idx, trial_run_id),
    FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
);

-- TRIAL_RUN to ARTIFACT relationship
CREATE TABLE TRIAL_RUN_ARTIFACT (
    trial_run_id INT,
    artifact_id INT,
    PRIMARY KEY (trial_run_id, artifact_id),
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id),
    FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
);

-- BATCH to METRIC relationship
CREATE TABLE BATCH_METRIC (
    batch_idx INT,
    epoch_idx INT,
    trial_run_id INT,
    metric_id INT,
    PRIMARY KEY (batch_idx, epoch_idx, trial_run_id, metric_id),
    FOREIGN KEY (batch_idx, epoch_idx, trial_run_id) REFERENCES BATCH(idx, epoch_idx, trial_run_id),
    FOREIGN KEY (metric_id) REFERENCES METRIC(id)
);

-- BATCH to ARTIFACT relationship
CREATE TABLE BATCH_ARTIFACT (
    batch_idx INT,
    epoch_idx INT,
    trial_run_id INT,
    artifact_id INT,
    PRIMARY KEY (batch_idx, epoch_idx, trial_run_id, artifact_id),
    FOREIGN KEY (batch_idx, epoch_idx, trial_run_id) REFERENCES BATCH(idx, epoch_idx, trial_run_id),
    FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
);
