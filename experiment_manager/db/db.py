"""Database initialization and utilities."""
import os
import sqlite3
import mysql.connector
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

# SQL statements for creating tables
MYSQL_TABLES = {
    "SCHEMA_VERSION": """
    CREATE TABLE IF NOT EXISTS SCHEMA_VERSION (
        id INT PRIMARY KEY AUTO_INCREMENT,
        version VARCHAR(20) NOT NULL UNIQUE,
        migration_name VARCHAR(255) NOT NULL,
        description TEXT,
        applied_at DATETIME NOT NULL,
        rollback_script TEXT
    )
    """,
    "EXPERIMENT": """
    CREATE TABLE IF NOT EXISTS EXPERIMENT (
        id INT PRIMARY KEY AUTO_INCREMENT,
        title VARCHAR(255) NOT NULL,
        `desc` TEXT,
        start_time DATETIME NOT NULL,
        update_time DATETIME NOT NULL
    )
    """,
    "TRIAL": """
    CREATE TABLE IF NOT EXISTS TRIAL (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL,
        experiment_id INT NOT NULL,
        start_time DATETIME NOT NULL,
        update_time DATETIME NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES EXPERIMENT(id)
    )
    """,
    "TRIAL_RUN": """
    CREATE TABLE IF NOT EXISTS TRIAL_RUN (
        id INT PRIMARY KEY AUTO_INCREMENT,
        trial_id INT NOT NULL,
        status VARCHAR(50) NOT NULL,
        start_time DATETIME NOT NULL,
        update_time DATETIME NOT NULL,
        FOREIGN KEY (trial_id) REFERENCES TRIAL(id)
    )
    """,
    "RESULTS": """
    CREATE TABLE IF NOT EXISTS RESULTS (
        trial_run_id INT PRIMARY KEY,
        time DATETIME NOT NULL,
        FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
    )
    """,
    "EPOCH": """
    CREATE TABLE IF NOT EXISTS EPOCH (
        idx INT,
        trial_run_id INT,
        time DATETIME NOT NULL,
        PRIMARY KEY (idx, trial_run_id),
        FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
    )
    """,
    "BATCH": """
    CREATE TABLE IF NOT EXISTS BATCH (
        idx INT,
        epoch_idx INT,
        trial_run_id INT,
        time DATETIME NOT NULL,
        PRIMARY KEY (idx, epoch_idx, trial_run_id),
        FOREIGN KEY (epoch_idx, trial_run_id) REFERENCES EPOCH(idx, trial_run_id)
    )
    """,
    "METRIC": """
    CREATE TABLE IF NOT EXISTS METRIC (
        id INT PRIMARY KEY AUTO_INCREMENT,
        type VARCHAR(50) NOT NULL,
        total_val FLOAT NOT NULL,
        per_label_val JSON
    )
    """,
    "ARTIFACT": """
    CREATE TABLE IF NOT EXISTS ARTIFACT (
        id INT PRIMARY KEY AUTO_INCREMENT,
        type VARCHAR(50) NOT NULL,
        loc VARCHAR(255) NOT NULL
    )
    """,
    "EXPERIMENT_ARTIFACT": """
    CREATE TABLE IF NOT EXISTS EXPERIMENT_ARTIFACT (
        experiment_id INT,
        artifact_id INT,
        PRIMARY KEY (experiment_id, artifact_id),
        FOREIGN KEY (experiment_id) REFERENCES EXPERIMENT(id),
        FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
    )
    """,
    "TRIAL_ARTIFACT": """
    CREATE TABLE IF NOT EXISTS TRIAL_ARTIFACT (
        trial_id INT,
        artifact_id INT,
        PRIMARY KEY (trial_id, artifact_id),
        FOREIGN KEY (trial_id) REFERENCES TRIAL(id),
        FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
    )
    """,
    "RESULTS_METRIC": """
    CREATE TABLE IF NOT EXISTS RESULTS_METRIC (
        results_id INT,
        metric_id INT,
        PRIMARY KEY (results_id, metric_id),
        FOREIGN KEY (results_id) REFERENCES RESULTS(trial_run_id),
        FOREIGN KEY (metric_id) REFERENCES METRIC(id)
    )
    """,
    "RESULTS_ARTIFACT": """
    CREATE TABLE IF NOT EXISTS RESULTS_ARTIFACT (
        results_id INT,
        artifact_id INT,
        PRIMARY KEY (results_id, artifact_id),
        FOREIGN KEY (results_id) REFERENCES RESULTS(trial_run_id),
        FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
    )
    """,
    "EPOCH_METRIC": """
    CREATE TABLE IF NOT EXISTS EPOCH_METRIC (
        epoch_idx INT,
        epoch_trial_run_id INT,
        metric_id INT,
        PRIMARY KEY (epoch_idx, epoch_trial_run_id, metric_id),
        FOREIGN KEY (epoch_idx, epoch_trial_run_id) REFERENCES EPOCH(idx, trial_run_id),
        FOREIGN KEY (metric_id) REFERENCES METRIC(id)
    )
    """,
    "EPOCH_ARTIFACT": """
    CREATE TABLE IF NOT EXISTS EPOCH_ARTIFACT (
        epoch_idx INT,
        epoch_trial_run_id INT,
        artifact_id INT,
        PRIMARY KEY (epoch_idx, epoch_trial_run_id, artifact_id),
        FOREIGN KEY (epoch_idx, epoch_trial_run_id) REFERENCES EPOCH(idx, trial_run_id),
        FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
    )
    """,
    "TRIAL_RUN_ARTIFACT": """
    CREATE TABLE IF NOT EXISTS TRIAL_RUN_ARTIFACT (
        trial_run_id INT,
        artifact_id INT,
        PRIMARY KEY (trial_run_id, artifact_id),
        FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id),
        FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
    )
    """,
    "BATCH_METRIC": """
    CREATE TABLE IF NOT EXISTS BATCH_METRIC (
        batch_idx INT,
        epoch_idx INT,
        trial_run_id INT,
        metric_id INT,
        PRIMARY KEY (batch_idx, epoch_idx, trial_run_id, metric_id),
        FOREIGN KEY (batch_idx, epoch_idx, trial_run_id) REFERENCES BATCH(idx, epoch_idx, trial_run_id),
        FOREIGN KEY (metric_id) REFERENCES METRIC(id)
    )
    """,
    "BATCH_ARTIFACT": """
    CREATE TABLE IF NOT EXISTS BATCH_ARTIFACT (
        batch_idx INT,
        epoch_idx INT,
        trial_run_id INT,
        artifact_id INT,
        PRIMARY KEY (batch_idx, epoch_idx, trial_run_id, artifact_id),
        FOREIGN KEY (batch_idx, epoch_idx, trial_run_id) REFERENCES BATCH(idx, epoch_idx, trial_run_id),
        FOREIGN KEY (artifact_id) REFERENCES ARTIFACT(id)
    )
    """
}

# SQLite version of the tables (replacing MySQL-specific syntax)
SQLITE_TABLES = {
    name: sql.replace("INT PRIMARY KEY AUTO_INCREMENT", "INTEGER PRIMARY KEY AUTOINCREMENT")
                   .replace("DATETIME", "TEXT")
                   .replace("`desc`", "desc")
                   .replace("JSON", "TEXT")
    for name, sql in MYSQL_TABLES.items()
}

def dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    """Convert SQLite row to dictionary."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def init_sqlite_db(db_path: Union[str, Path], recreate: bool = False, readonly: bool = False) -> sqlite3.Connection:
    """Initialize SQLite database with all tables.
    
    Args:
        db_path: Path to SQLite database file
        recreate: If True, delete existing database file
        readonly: If True, open database in readonly mode
    Returns:
        SQLite connection object
    """
    db_path = Path(db_path)
    
    if recreate and db_path.exists():
        os.remove(db_path)
        
    # Create parent directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    
    if not readonly:
        # Create tables in order (due to foreign key constraints)
        for table_name, create_sql in SQLITE_TABLES.items():
            try:
                cursor.execute(create_sql)
            except sqlite3.OperationalError as e:
                print(f"Error creating table {table_name}: {e}")
                raise
        conn.commit()
    return conn

def init_mysql_db(host: str, user: str, password: str, database: str, 
                 recreate: bool = False) -> mysql.connector.MySQLConnection:
    """Initialize MySQL database with all tables.
    
    Args:
        host: Database host
        user: Database user
        password: Database password
        database: Database name
        recreate: If True, drop and recreate database
        
    Returns:
        MySQL connection object
    """
    # First connect without database to create it
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password
    )
    cursor = conn.cursor(dictionary=True)
    
    if recreate:
        cursor.execute(f"DROP DATABASE IF EXISTS {database}")
        
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
    cursor.execute(f"USE {database}")
    
    # Create tables in order (due to foreign key constraints)
    for table_name, create_sql in MYSQL_TABLES.items():
        try:
            cursor.execute(create_sql)
        except mysql.connector.Error as e:
            print(f"Error creating table {table_name}: {e}")
            raise
    
    conn.commit()
    return conn
