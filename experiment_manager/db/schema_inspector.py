"""Database schema introspection utilities for schema comparison and diff generation."""
import logging
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from experiment_manager.db.manager import DatabaseManager, DatabaseError

logger = logging.getLogger(__name__)

class ColumnType(Enum):
    """Standard column types across database systems."""
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    TEXT = "TEXT"
    DATETIME = "DATETIME"
    JSON = "JSON"
    BOOLEAN = "BOOLEAN"
    BLOB = "BLOB"

class ConstraintType(Enum):
    """Database constraint types."""
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    UNIQUE = "UNIQUE"
    NOT_NULL = "NOT_NULL"
    CHECK = "CHECK"
    DEFAULT = "DEFAULT"

@dataclass
class ColumnInfo:
    """Represents a database column with all its properties."""
    name: str
    data_type: str
    normalized_type: ColumnType
    is_nullable: bool
    default_value: Optional[str] = None
    is_primary_key: bool = False
    is_auto_increment: bool = False
    character_maximum_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    collation_name: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class IndexInfo:
    """Represents a database index."""
    name: str
    table_name: str
    columns: List[str]
    is_unique: bool
    is_primary: bool
    index_type: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class ForeignKeyInfo:
    """Represents a foreign key constraint."""
    name: str
    table_name: str
    column_name: str
    referenced_table: str
    referenced_column: str
    on_delete: Optional[str] = None
    on_update: Optional[str] = None

@dataclass
class CheckConstraintInfo:
    """Represents a check constraint."""
    name: str
    table_name: str
    check_clause: str

@dataclass
class TableInfo:
    """Represents a complete table schema."""
    name: str
    columns: List[ColumnInfo]
    indexes: List[IndexInfo]
    foreign_keys: List[ForeignKeyInfo]
    check_constraints: List[CheckConstraintInfo]
    engine: Optional[str] = None
    charset: Optional[str] = None
    collation: Optional[str] = None
    comment: Optional[str] = None
    row_count: Optional[int] = None
    data_size_bytes: Optional[int] = None

@dataclass
class DatabaseSchema:
    """Represents a complete database schema."""
    database_name: str
    database_type: str  # 'sqlite' or 'mysql'
    version: Optional[str]
    tables: List[TableInfo]
    schema_version: Optional[str] = None
    extracted_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class SchemaInspector:
    """Inspects database schemas for comparison and analysis."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize schema inspector.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager
        self.database_type = "sqlite" if db_manager.use_sqlite else "mysql"
    
    def extract_full_schema(self, include_data_stats: bool = True) -> DatabaseSchema:
        """Extract complete database schema information.
        
        Args:
            include_data_stats: Whether to include row counts and data sizes
            
        Returns:
            DatabaseSchema: Complete schema information
        """
        try:
            tables = []
            table_names = self._get_table_names()
            
            for table_name in table_names:
                table_info = self._extract_table_schema(table_name, include_data_stats)
                tables.append(table_info)
            
            # Get current schema version
            schema_version = None
            try:
                current_version = self.db_manager.get_current_schema_version()
                schema_version = current_version.version if current_version else None
            except Exception:
                pass  # Schema versioning might not be initialized
            
            # Get database name
            database_name = self._get_database_name()
            
            return DatabaseSchema(
                database_name=database_name,
                database_type=self.database_type,
                version=self._get_database_version(),
                tables=tables,
                schema_version=schema_version,
                extracted_at=datetime.now(),
                metadata=self._collect_database_metadata()
            )
        
        except Exception as e:
            raise DatabaseError(f"Failed to extract schema: {e}") from e
    
    def _get_table_names(self) -> List[str]:
        """Get list of all table names in the database."""
        if self.database_type == "sqlite":
            query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        else:  # MySQL
            query = "SHOW TABLES"
        
        cursor = self.db_manager._execute_query(query)
        rows = cursor.fetchall()
        
        if self.database_type == "sqlite":
            return [row["name"] for row in rows]
        else:
            # MySQL returns table names in the first column
            return [list(row.values())[0] for row in rows]
    
    def _extract_table_schema(self, table_name: str, include_data_stats: bool = True) -> TableInfo:
        """Extract schema information for a specific table.
        
        Args:
            table_name: Name of the table to analyze
            include_data_stats: Whether to include row counts and data sizes
            
        Returns:
            TableInfo: Complete table schema information
        """
        columns = self._get_table_columns(table_name)
        indexes = self._get_table_indexes(table_name)
        foreign_keys = self._get_table_foreign_keys(table_name)
        check_constraints = self._get_table_check_constraints(table_name)
        
        # Get table metadata
        engine = None
        charset = None
        collation = None
        comment = None
        row_count = None
        data_size_bytes = None
        
        if self.database_type == "mysql":
            table_meta = self._get_mysql_table_metadata(table_name)
            engine = table_meta.get("engine")
            charset = table_meta.get("charset")
            collation = table_meta.get("collation")
            comment = table_meta.get("comment")
        
        if include_data_stats:
            row_count = self._get_table_row_count(table_name)
            data_size_bytes = self._get_table_size(table_name)
        
        return TableInfo(
            name=table_name,
            columns=columns,
            indexes=indexes,
            foreign_keys=foreign_keys,
            check_constraints=check_constraints,
            engine=engine,
            charset=charset,
            collation=collation,
            comment=comment,
            row_count=row_count,
            data_size_bytes=data_size_bytes
        )
    
    def _get_table_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get column information for a table."""
        if self.database_type == "sqlite":
            return self._get_sqlite_columns(table_name)
        else:
            return self._get_mysql_columns(table_name)
    
    def _get_sqlite_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get column information for SQLite table."""
        query = f"PRAGMA table_info({table_name})"
        cursor = self.db_manager._execute_query(query)
        columns_data = cursor.fetchall()
        
        # Get primary key info
        pk_query = f"PRAGMA table_info({table_name})"
        pk_cursor = self.db_manager._execute_query(pk_query)
        pk_info = pk_cursor.fetchall()
        primary_keys = {col["name"] for col in pk_info if col["pk"]}
        
        columns = []
        for col_data in columns_data:
            # Normalize type
            raw_type = col_data["type"].upper()
            normalized_type = self._normalize_column_type(raw_type)
            
            # Check for auto increment
            is_auto_increment = (
                col_data["pk"] and 
                "INTEGER" in raw_type.upper() and 
                col_data["name"] in primary_keys
            )
            
            column = ColumnInfo(
                name=col_data["name"],
                data_type=col_data["type"],
                normalized_type=normalized_type,
                is_nullable=not bool(col_data["notnull"]),
                default_value=col_data["dflt_value"],
                is_primary_key=bool(col_data["pk"]),
                is_auto_increment=is_auto_increment
            )
            columns.append(column)
        
        return columns
    
    def _get_mysql_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get column information for MySQL table."""
        query = """
        SELECT 
            COLUMN_NAME as name,
            DATA_TYPE as data_type,
            COLUMN_TYPE as full_type,
            IS_NULLABLE as is_nullable,
            COLUMN_DEFAULT as default_value,
            COLUMN_KEY as key_type,
            EXTRA as extra,
            CHARACTER_MAXIMUM_LENGTH as char_max_length,
            NUMERIC_PRECISION as numeric_precision,
            NUMERIC_SCALE as numeric_scale,
            COLLATION_NAME as collation_name,
            COLUMN_COMMENT as comment
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        
        cursor = self.db_manager._execute_query(query, (table_name,))
        columns_data = cursor.fetchall()
        
        columns = []
        for col_data in columns_data:
            # Normalize type
            normalized_type = self._normalize_column_type(col_data["data_type"])
            
            column = ColumnInfo(
                name=col_data["name"],
                data_type=col_data["full_type"],
                normalized_type=normalized_type,
                is_nullable=col_data["is_nullable"] == "YES",
                default_value=col_data["default_value"],
                is_primary_key=col_data["key_type"] == "PRI",
                is_auto_increment="auto_increment" in (col_data["extra"] or "").lower(),
                character_maximum_length=col_data["char_max_length"],
                numeric_precision=col_data["numeric_precision"],
                numeric_scale=col_data["numeric_scale"],
                collation_name=col_data["collation_name"],
                comment=col_data["comment"]
            )
            columns.append(column)
        
        return columns
    
    def _get_table_indexes(self, table_name: str) -> List[IndexInfo]:
        """Get index information for a table."""
        if self.database_type == "sqlite":
            return self._get_sqlite_indexes(table_name)
        else:
            return self._get_mysql_indexes(table_name)
    
    def _get_sqlite_indexes(self, table_name: str) -> List[IndexInfo]:
        """Get index information for SQLite table."""
        query = f"PRAGMA index_list({table_name})"
        cursor = self.db_manager._execute_query(query)
        index_list = cursor.fetchall()
        
        indexes = []
        for index_data in index_list:
            index_name = index_data["name"]
            
            # Get index columns
            col_query = f"PRAGMA index_info({index_name})"
            col_cursor = self.db_manager._execute_query(col_query)
            col_data = col_cursor.fetchall()
            
            columns = [col["name"] for col in col_data]
            
            index = IndexInfo(
                name=index_name,
                table_name=table_name,
                columns=columns,
                is_unique=bool(index_data["unique"]),
                is_primary=index_data["origin"] == "pk"
            )
            indexes.append(index)
        
        return indexes
    
    def _get_mysql_indexes(self, table_name: str) -> List[IndexInfo]:
        """Get index information for MySQL table."""
        query = """
        SELECT 
            INDEX_NAME as name,
            NON_UNIQUE as non_unique,
            COLUMN_NAME as column_name,
            INDEX_TYPE as index_type,
            INDEX_COMMENT as comment
        FROM INFORMATION_SCHEMA.STATISTICS 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        ORDER BY INDEX_NAME, SEQ_IN_INDEX
        """
        
        cursor = self.db_manager._execute_query(query, (table_name,))
        index_data = cursor.fetchall()
        
        # Group by index name
        indexes_dict = {}
        for row in index_data:
            index_name = row["name"]
            if index_name not in indexes_dict:
                indexes_dict[index_name] = {
                    "columns": [],
                    "is_unique": row["non_unique"] == 0,
                    "is_primary": index_name == "PRIMARY",
                    "index_type": row["index_type"],
                    "comment": row["comment"]
                }
            indexes_dict[index_name]["columns"].append(row["column_name"])
        
        indexes = []
        for index_name, index_info in indexes_dict.items():
            index = IndexInfo(
                name=index_name,
                table_name=table_name,
                columns=index_info["columns"],
                is_unique=index_info["is_unique"],
                is_primary=index_info["is_primary"],
                index_type=index_info["index_type"],
                comment=index_info["comment"]
            )
            indexes.append(index)
        
        return indexes
    
    def _get_table_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Get foreign key information for a table."""
        if self.database_type == "sqlite":
            return self._get_sqlite_foreign_keys(table_name)
        else:
            return self._get_mysql_foreign_keys(table_name)
    
    def _get_sqlite_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Get foreign key information for SQLite table."""
        query = f"PRAGMA foreign_key_list({table_name})"
        cursor = self.db_manager._execute_query(query)
        fk_data = cursor.fetchall()
        
        foreign_keys = []
        for fk in fk_data:
            foreign_key = ForeignKeyInfo(
                name=f"fk_{table_name}_{fk['from']}_{fk['table']}_{fk['to']}",
                table_name=table_name,
                column_name=fk["from"],
                referenced_table=fk["table"],
                referenced_column=fk["to"],
                on_delete=fk["on_delete"],
                on_update=fk["on_update"]
            )
            foreign_keys.append(foreign_key)
        
        return foreign_keys
    
    def _get_mysql_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Get foreign key information for MySQL table."""
        query = """
        SELECT 
            CONSTRAINT_NAME as name,
            COLUMN_NAME as column_name,
            REFERENCED_TABLE_NAME as referenced_table,
            REFERENCED_COLUMN_NAME as referenced_column,
            DELETE_RULE as on_delete,
            UPDATE_RULE as on_update
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
        JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc 
        ON kcu.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
        WHERE kcu.TABLE_SCHEMA = DATABASE() 
        AND kcu.TABLE_NAME = %s 
        AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
        """
        
        cursor = self.db_manager._execute_query(query, (table_name,))
        fk_data = cursor.fetchall()
        
        foreign_keys = []
        for fk in fk_data:
            foreign_key = ForeignKeyInfo(
                name=fk["name"],
                table_name=table_name,
                column_name=fk["column_name"],
                referenced_table=fk["referenced_table"],
                referenced_column=fk["referenced_column"],
                on_delete=fk["on_delete"],
                on_update=fk["on_update"]
            )
            foreign_keys.append(foreign_key)
        
        return foreign_keys
    
    def _get_table_check_constraints(self, table_name: str) -> List[CheckConstraintInfo]:
        """Get check constraint information for a table."""
        if self.database_type == "sqlite":
            # SQLite doesn't provide easy access to check constraints via PRAGMA
            return []
        else:
            return self._get_mysql_check_constraints(table_name)
    
    def _get_mysql_check_constraints(self, table_name: str) -> List[CheckConstraintInfo]:
        """Get check constraint information for MySQL table."""
        query = """
        SELECT 
            CONSTRAINT_NAME as name,
            CHECK_CLAUSE as check_clause
        FROM INFORMATION_SCHEMA.CHECK_CONSTRAINTS 
        WHERE CONSTRAINT_SCHEMA = DATABASE() 
        AND TABLE_NAME = %s
        """
        
        cursor = self.db_manager._execute_query(query, (table_name,))
        check_data = cursor.fetchall()
        
        constraints = []
        for constraint in check_data:
            check_constraint = CheckConstraintInfo(
                name=constraint["name"],
                table_name=table_name,
                check_clause=constraint["check_clause"]
            )
            constraints.append(check_constraint)
        
        return constraints
    
    def _normalize_column_type(self, raw_type: str) -> ColumnType:
        """Normalize database-specific column types to standard types."""
        raw_type = raw_type.upper()
        
        if any(t in raw_type for t in ["INT", "BIGINT", "SMALLINT", "TINYINT", "SERIAL"]):
            return ColumnType.INTEGER
        elif any(t in raw_type for t in ["FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC"]):
            return ColumnType.FLOAT
        elif any(t in raw_type for t in ["TEXT", "VARCHAR", "CHAR", "STRING"]):
            return ColumnType.TEXT
        elif any(t in raw_type for t in ["DATETIME", "TIMESTAMP", "DATE", "TIME"]):
            return ColumnType.DATETIME
        elif "JSON" in raw_type:
            return ColumnType.JSON
        elif any(t in raw_type for t in ["BOOL", "BOOLEAN"]):
            return ColumnType.BOOLEAN
        elif any(t in raw_type for t in ["BLOB", "BINARY", "VARBINARY"]):
            return ColumnType.BLOB
        else:
            return ColumnType.TEXT  # Default fallback
    
    def _get_database_name(self) -> str:
        """Get the database name."""
        if self.database_type == "sqlite":
            if hasattr(self.db_manager, 'database_path') and self.db_manager.database_path:
                from pathlib import Path
                return Path(self.db_manager.database_path).stem
            else:
                return "sqlite_database"
        else:
            cursor = self.db_manager._execute_query("SELECT DATABASE() as name")
            result = cursor.fetchone()
            return result["name"] if result else "mysql_database"
    
    def _get_database_version(self) -> str:
        """Get the database version."""
        if self.database_type == "sqlite":
            cursor = self.db_manager._execute_query("SELECT sqlite_version() as version")
            result = cursor.fetchone()
            return f"SQLite {result['version']}"
        else:
            cursor = self.db_manager._execute_query("SELECT VERSION() as version")
            result = cursor.fetchone()
            return f"MySQL {result['version']}"
    
    def _get_mysql_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get MySQL-specific table metadata."""
        query = """
        SELECT 
            ENGINE as engine,
            TABLE_COLLATION as collation,
            TABLE_COMMENT as comment
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """
        
        cursor = self.db_manager._execute_query(query, (table_name,))
        result = cursor.fetchone()
        
        if result:
            metadata = {
                "engine": result["engine"],
                "collation": result["collation"],
                "comment": result["comment"]
            }
            # Extract charset from collation
            if result["collation"]:
                metadata["charset"] = result["collation"].split("_")[0]
            return metadata
        
        return {}
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        try:
            cursor = self.db_manager._execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            result = cursor.fetchone()
            return result["count"] if result else 0
        except Exception as e:
            logger.warning(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def _get_table_size(self, table_name: str) -> Optional[int]:
        """Get table size in bytes."""
        if self.database_type == "mysql":
            query = """
            SELECT (DATA_LENGTH + INDEX_LENGTH) as size_bytes
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
            """
            try:
                cursor = self.db_manager._execute_query(query, (table_name,))
                result = cursor.fetchone()
                return result["size_bytes"] if result else None
            except Exception:
                return None
        else:
            # SQLite doesn't provide easy table size info
            return None
    
    def _collect_database_metadata(self) -> Dict[str, Any]:
        """Collect additional database metadata."""
        metadata = {
            "extraction_time": datetime.now().isoformat(),
            "database_type": self.database_type
        }
        
        try:
            # Get total table count
            cursor = self.db_manager._execute_query(
                "SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'"
                if self.database_type == "sqlite"
                else "SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE()"
            )
            result = cursor.fetchone()
            metadata["total_tables"] = result["count"] if result else 0
            
            # Get schema version if available
            try:
                current_version = self.db_manager.get_current_schema_version()
                if current_version:
                    metadata["current_schema_version"] = current_version.version
                    metadata["last_migration"] = current_version.migration_name
                    metadata["migration_applied_at"] = current_version.applied_at.isoformat()
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"Failed to collect some metadata: {e}")
        
        return metadata
    
    def save_schema_to_file(self, schema: DatabaseSchema, output_path: str) -> None:
        """Save schema to JSON file.
        
        Args:
            schema: Database schema to save
            output_path: Path to output file
        """
        from pathlib import Path
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for JSON serialization
        schema_dict = asdict(schema)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, default=str)
        
        logger.info(f"Schema saved to {output_path}")
    
    def load_schema_from_file(self, input_path: str) -> DatabaseSchema:
        """Load schema from JSON file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            DatabaseSchema: Loaded schema
        """
        from pathlib import Path
        
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Schema file not found: {input_path}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to dataclass instances
        # This is a simplified approach - in production you might want more robust deserialization
        return DatabaseSchema(**data) 