"""Schema comparison and diff generation utilities for database schema analysis."""
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import json

from experiment_manager.db.schema_inspector import (
    DatabaseSchema, TableInfo, ColumnInfo, IndexInfo, ForeignKeyInfo, 
    CheckConstraintInfo, SchemaInspector, ColumnType
)
from experiment_manager.db.manager import DatabaseManager

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of schema changes."""
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"
    UNCHANGED = "UNCHANGED"

class ImpactLevel(Enum):
    """Impact levels for schema changes."""
    COSMETIC = 0      # Non-functional changes (comments, etc.)
    MINOR = 1         # Minor change with minimal impact
    MAJOR = 2         # Significant change that may affect operations
    BREAKING = 3      # Requires data migration or breaks compatibility
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

@dataclass
class ColumnDiff:
    """Represents differences in a column."""
    column_name: str
    change_type: ChangeType
    old_column: Optional[ColumnInfo] = None
    new_column: Optional[ColumnInfo] = None
    impact_level: ImpactLevel = ImpactLevel.MINOR
    impact_description: str = ""
    migration_notes: str = ""

@dataclass
class IndexDiff:
    """Represents differences in an index."""
    index_name: str
    change_type: ChangeType
    old_index: Optional[IndexInfo] = None
    new_index: Optional[IndexInfo] = None
    impact_level: ImpactLevel = ImpactLevel.MINOR
    impact_description: str = ""

@dataclass
class ForeignKeyDiff:
    """Represents differences in a foreign key."""
    fk_name: str
    change_type: ChangeType
    old_fk: Optional[ForeignKeyInfo] = None
    new_fk: Optional[ForeignKeyInfo] = None
    impact_level: ImpactLevel = ImpactLevel.MAJOR
    impact_description: str = ""

@dataclass
class TableDiff:
    """Represents differences in a table."""
    table_name: str
    change_type: ChangeType
    old_table: Optional[TableInfo] = None
    new_table: Optional[TableInfo] = None
    column_diffs: List[ColumnDiff] = None
    index_diffs: List[IndexDiff] = None
    foreign_key_diffs: List[ForeignKeyDiff] = None
    impact_level: ImpactLevel = ImpactLevel.MINOR
    impact_description: str = ""
    migration_complexity: str = "LOW"  # LOW, MEDIUM, HIGH
    
    def __post_init__(self):
        if self.column_diffs is None:
            self.column_diffs = []
        if self.index_diffs is None:
            self.index_diffs = []
        if self.foreign_key_diffs is None:
            self.foreign_key_diffs = []

@dataclass
class SchemaDiff:
    """Represents complete schema differences."""
    source_schema: DatabaseSchema
    target_schema: DatabaseSchema
    table_diffs: List[TableDiff]
    overall_impact: ImpactLevel
    migration_strategy: str
    risk_assessment: str
    estimated_downtime: str
    rollback_complexity: str
    dependency_changes: List[Dict[str, Any]]
    generated_at: datetime
    summary: Dict[str, int]
    
    def __post_init__(self):
        if not self.summary:
            self.summary = self._calculate_summary()
    
    def _calculate_summary(self) -> Dict[str, int]:
        """Calculate summary statistics of changes."""
        summary = {
            "total_tables": len(self.table_diffs),
            "added_tables": 0,
            "removed_tables": 0,
            "modified_tables": 0,
            "total_columns": 0,
            "added_columns": 0,
            "removed_columns": 0,
            "modified_columns": 0,
            "total_indexes": 0,
            "added_indexes": 0,
            "removed_indexes": 0,
            "modified_indexes": 0,
            "total_foreign_keys": 0,
            "added_foreign_keys": 0,
            "removed_foreign_keys": 0,
            "modified_foreign_keys": 0,
            "breaking_changes": 0,
            "major_changes": 0,
            "minor_changes": 0
        }
        
        for table_diff in self.table_diffs:
            if table_diff.change_type == ChangeType.ADDED:
                summary["added_tables"] += 1
                # Count all columns in new table as added
                if table_diff.new_table:
                    summary["added_columns"] += len(table_diff.new_table.columns)
            elif table_diff.change_type == ChangeType.REMOVED:
                summary["removed_tables"] += 1
                # Count all columns in removed table as removed
                if table_diff.old_table:
                    summary["removed_columns"] += len(table_diff.old_table.columns)
            elif table_diff.change_type == ChangeType.MODIFIED:
                summary["modified_tables"] += 1
            
            # Count column changes (for modified tables)
            for col_diff in table_diff.column_diffs:
                summary["total_columns"] += 1
                if col_diff.change_type == ChangeType.ADDED:
                    summary["added_columns"] += 1
                elif col_diff.change_type == ChangeType.REMOVED:
                    summary["removed_columns"] += 1
                elif col_diff.change_type == ChangeType.MODIFIED:
                    summary["modified_columns"] += 1
                
                # Count impact levels
                if col_diff.impact_level == ImpactLevel.BREAKING:
                    summary["breaking_changes"] += 1
                elif col_diff.impact_level == ImpactLevel.MAJOR:
                    summary["major_changes"] += 1
                elif col_diff.impact_level == ImpactLevel.MINOR:
                    summary["minor_changes"] += 1
            
            # Count index changes
            for idx_diff in table_diff.index_diffs:
                summary["total_indexes"] += 1
                if idx_diff.change_type == ChangeType.ADDED:
                    summary["added_indexes"] += 1
                elif idx_diff.change_type == ChangeType.REMOVED:
                    summary["removed_indexes"] += 1
                elif idx_diff.change_type == ChangeType.MODIFIED:
                    summary["modified_indexes"] += 1
            
            # Count foreign key changes
            for fk_diff in table_diff.foreign_key_diffs:
                summary["total_foreign_keys"] += 1
                if fk_diff.change_type == ChangeType.ADDED:
                    summary["added_foreign_keys"] += 1
                elif fk_diff.change_type == ChangeType.REMOVED:
                    summary["removed_foreign_keys"] += 1
                elif fk_diff.change_type == ChangeType.MODIFIED:
                    summary["modified_foreign_keys"] += 1
        
        return summary

class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Enum objects."""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name  # Use enum name instead of value for consistency
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class SchemaComparator:
    """Compares database schemas and generates comprehensive diff reports."""
    
    def __init__(self):
        """Initialize schema comparator."""
        self.impact_rules = self._initialize_impact_rules()
    
    def compare_schemas(self, source_schema: DatabaseSchema, 
                       target_schema: DatabaseSchema) -> SchemaDiff:
        """Compare two database schemas and generate a comprehensive diff.
        
        Args:
            source_schema: Original schema
            target_schema: New schema to compare against
            
        Returns:
            SchemaDiff: Comprehensive diff report
        """
        logger.info(f"Comparing schemas: {source_schema.database_name} -> {target_schema.database_name}")
        
        # Compare tables
        table_diffs = self._compare_tables(source_schema.tables, target_schema.tables)
        
        # Analyze dependencies
        dependency_changes = self._analyze_dependency_changes(table_diffs)
        
        # Assess overall impact
        overall_impact = self._assess_overall_impact(table_diffs)
        
        # Generate migration strategy
        migration_strategy = self._generate_migration_strategy(table_diffs, overall_impact)
        
        # Assess risks
        risk_assessment = self._assess_risks(table_diffs, overall_impact)
        
        # Estimate downtime
        estimated_downtime = self._estimate_downtime(table_diffs, source_schema)
        
        # Assess rollback complexity
        rollback_complexity = self._assess_rollback_complexity(table_diffs)
        
        return SchemaDiff(
            source_schema=source_schema,
            target_schema=target_schema,
            table_diffs=table_diffs,
            overall_impact=overall_impact,
            migration_strategy=migration_strategy,
            risk_assessment=risk_assessment,
            estimated_downtime=estimated_downtime,
            rollback_complexity=rollback_complexity,
            dependency_changes=dependency_changes,
            generated_at=datetime.now(),
            summary={}  # Will be calculated in __post_init__
        )
    
    def _compare_tables(self, source_tables: List[TableInfo], 
                       target_tables: List[TableInfo]) -> List[TableDiff]:
        """Compare tables between two schemas."""
        source_table_map = {t.name: t for t in source_tables}
        target_table_map = {t.name: t for t in target_tables}
        
        all_table_names = set(source_table_map.keys()) | set(target_table_map.keys())
        table_diffs = []
        
        for table_name in sorted(all_table_names):
            source_table = source_table_map.get(table_name)
            target_table = target_table_map.get(table_name)
            
            if source_table and target_table:
                # Table exists in both - compare details
                table_diff = self._compare_table_details(source_table, target_table)
            elif target_table and not source_table:
                # New table
                table_diff = TableDiff(
                    table_name=table_name,
                    change_type=ChangeType.ADDED,
                    new_table=target_table,
                    impact_level=ImpactLevel.MINOR,
                    impact_description=f"New table '{table_name}' added",
                    migration_complexity="LOW"
                )
            else:
                # Removed table
                table_diff = TableDiff(
                    table_name=table_name,
                    change_type=ChangeType.REMOVED,
                    old_table=source_table,
                    impact_level=ImpactLevel.BREAKING,
                    impact_description=f"Table '{table_name}' removed - potential data loss",
                    migration_complexity="HIGH"
                )
            
            table_diffs.append(table_diff)
        
        return table_diffs
    
    def _compare_table_details(self, source_table: TableInfo, 
                              target_table: TableInfo) -> TableDiff:
        """Compare detailed differences within a table."""
        column_diffs = self._compare_columns(source_table.columns, target_table.columns, source_table.name)
        index_diffs = self._compare_indexes(source_table.indexes, target_table.indexes)
        fk_diffs = self._compare_foreign_keys(source_table.foreign_keys, target_table.foreign_keys)
        
        # Determine overall table impact
        max_impact = ImpactLevel.COSMETIC
        impact_descriptions = []
        
        for diff_list in [column_diffs, index_diffs, fk_diffs]:
            for diff in diff_list:
                if hasattr(diff, 'impact_level'):
                    if diff.impact_level.value == "BREAKING" and max_impact.value != "BREAKING":
                        max_impact = ImpactLevel.BREAKING
                    elif diff.impact_level.value == "MAJOR" and max_impact.value in ["MINOR", "COSMETIC"]:
                        max_impact = ImpactLevel.MAJOR
                    elif diff.impact_level.value == "MINOR" and max_impact.value == "COSMETIC":
                        max_impact = ImpactLevel.MINOR
                
                if hasattr(diff, 'impact_description') and diff.impact_description:
                    impact_descriptions.append(diff.impact_description)
        
        # Determine migration complexity
        migration_complexity = "LOW"
        breaking_changes = sum(1 for cd in column_diffs if cd.impact_level == ImpactLevel.BREAKING)
        major_changes = sum(1 for cd in column_diffs if cd.impact_level == ImpactLevel.MAJOR)
        
        if breaking_changes > 0:
            migration_complexity = "HIGH"
        elif major_changes > 2:
            migration_complexity = "MEDIUM"
        
        change_type = ChangeType.UNCHANGED
        if column_diffs or index_diffs or fk_diffs:
            change_type = ChangeType.MODIFIED
        
        return TableDiff(
            table_name=source_table.name,
            change_type=change_type,
            old_table=source_table,
            new_table=target_table,
            column_diffs=column_diffs,
            index_diffs=index_diffs,
            foreign_key_diffs=fk_diffs,
            impact_level=max_impact,
            impact_description="; ".join(impact_descriptions) if impact_descriptions else "",
            migration_complexity=migration_complexity
        )
    
    def _compare_columns(self, source_columns: List[ColumnInfo], 
                        target_columns: List[ColumnInfo], table_name: str) -> List[ColumnDiff]:
        """Compare columns between two tables."""
        source_col_map = {c.name: c for c in source_columns}
        target_col_map = {c.name: c for c in target_columns}
        
        all_col_names = set(source_col_map.keys()) | set(target_col_map.keys())
        column_diffs = []
        
        for col_name in sorted(all_col_names):
            source_col = source_col_map.get(col_name)
            target_col = target_col_map.get(col_name)
            
            if source_col and target_col:
                # Column exists in both - check for modifications
                diff = self._compare_column_details(source_col, target_col, table_name)
                if diff:
                    column_diffs.append(diff)
            elif target_col and not source_col:
                # New column
                impact_level, impact_desc, migration_notes = self._assess_column_addition_impact(target_col, table_name)
                column_diffs.append(ColumnDiff(
                    column_name=col_name,
                    change_type=ChangeType.ADDED,
                    new_column=target_col,
                    impact_level=impact_level,
                    impact_description=impact_desc,
                    migration_notes=migration_notes
                ))
            else:
                # Removed column
                impact_level, impact_desc, migration_notes = self._assess_column_removal_impact(source_col, table_name)
                column_diffs.append(ColumnDiff(
                    column_name=col_name,
                    change_type=ChangeType.REMOVED,
                    old_column=source_col,
                    impact_level=impact_level,
                    impact_description=impact_desc,
                    migration_notes=migration_notes
                ))
        
        return column_diffs
    
    def _compare_column_details(self, source_col: ColumnInfo, target_col: ColumnInfo, 
                               table_name: str) -> Optional[ColumnDiff]:
        """Compare detailed differences between two columns."""
        differences = []
        impact_level = ImpactLevel.COSMETIC
        migration_notes = []
        
        # Check data type changes
        if source_col.normalized_type != target_col.normalized_type:
            differences.append(f"Type changed from {source_col.normalized_type.value} to {target_col.normalized_type.value}")
            impact_level = ImpactLevel.BREAKING
            migration_notes.append("Data type conversion required")
        
        # Check nullability changes
        if source_col.is_nullable != target_col.is_nullable:
            if target_col.is_nullable:
                differences.append("Column made nullable")
                impact_level = max(impact_level, ImpactLevel.MINOR)
            else:
                differences.append("Column made NOT NULL")
                impact_level = ImpactLevel.BREAKING
                migration_notes.append("Ensure no NULL values exist before migration")
        
        # Check primary key changes
        if source_col.is_primary_key != target_col.is_primary_key:
            if target_col.is_primary_key:
                differences.append("Column made primary key")
                impact_level = ImpactLevel.BREAKING
            else:
                differences.append("Primary key constraint removed")
                impact_level = ImpactLevel.BREAKING
        
        # Check default value changes
        if source_col.default_value != target_col.default_value:
            differences.append(f"Default value changed from '{source_col.default_value}' to '{target_col.default_value}'")
            impact_level = max(impact_level, ImpactLevel.MINOR)
        
        # Check auto increment changes
        if source_col.is_auto_increment != target_col.is_auto_increment:
            differences.append("Auto increment setting changed")
            impact_level = ImpactLevel.MAJOR
        
        # Check character length changes (for text columns)
        if (source_col.character_maximum_length != target_col.character_maximum_length and 
            source_col.character_maximum_length is not None and 
            target_col.character_maximum_length is not None):
            
            if target_col.character_maximum_length < source_col.character_maximum_length:
                differences.append(f"Character length reduced from {source_col.character_maximum_length} to {target_col.character_maximum_length}")
                impact_level = ImpactLevel.BREAKING
                migration_notes.append("Check for data truncation")
            else:
                differences.append(f"Character length increased from {source_col.character_maximum_length} to {target_col.character_maximum_length}")
                impact_level = max(impact_level, ImpactLevel.MINOR)
        
        if not differences:
            return None
        
        return ColumnDiff(
            column_name=source_col.name,
            change_type=ChangeType.MODIFIED,
            old_column=source_col,
            new_column=target_col,
            impact_level=impact_level,
            impact_description="; ".join(differences),
            migration_notes="; ".join(migration_notes)
        )
    
    def _compare_indexes(self, source_indexes: List[IndexInfo], 
                        target_indexes: List[IndexInfo]) -> List[IndexDiff]:
        """Compare indexes between two tables."""
        source_idx_map = {i.name: i for i in source_indexes}
        target_idx_map = {i.name: i for i in target_indexes}
        
        all_idx_names = set(source_idx_map.keys()) | set(target_idx_map.keys())
        index_diffs = []
        
        for idx_name in sorted(all_idx_names):
            source_idx = source_idx_map.get(idx_name)
            target_idx = target_idx_map.get(idx_name)
            
            if source_idx and target_idx:
                # Index exists in both - check for modifications
                if (source_idx.columns != target_idx.columns or 
                    source_idx.is_unique != target_idx.is_unique):
                    
                    impact_level = ImpactLevel.MAJOR if source_idx.is_unique != target_idx.is_unique else ImpactLevel.MINOR
                    impact_desc = f"Index '{idx_name}' modified"
                    
                    index_diffs.append(IndexDiff(
                        index_name=idx_name,
                        change_type=ChangeType.MODIFIED,
                        old_index=source_idx,
                        new_index=target_idx,
                        impact_level=impact_level,
                        impact_description=impact_desc
                    ))
            elif target_idx and not source_idx:
                # New index
                impact_level = ImpactLevel.MINOR
                impact_desc = f"New index '{idx_name}' added"
                
                index_diffs.append(IndexDiff(
                    index_name=idx_name,
                    change_type=ChangeType.ADDED,
                    new_index=target_idx,
                    impact_level=impact_level,
                    impact_description=impact_desc
                ))
            else:
                # Removed index
                impact_level = ImpactLevel.MAJOR if source_idx.is_unique or source_idx.is_primary else ImpactLevel.MINOR
                impact_desc = f"Index '{idx_name}' removed"
                
                index_diffs.append(IndexDiff(
                    index_name=idx_name,
                    change_type=ChangeType.REMOVED,
                    old_index=source_idx,
                    impact_level=impact_level,
                    impact_description=impact_desc
                ))
        
        return index_diffs
    
    def _compare_foreign_keys(self, source_fks: List[ForeignKeyInfo], 
                             target_fks: List[ForeignKeyInfo]) -> List[ForeignKeyDiff]:
        """Compare foreign keys between two tables."""
        source_fk_map = {f.name: f for f in source_fks}
        target_fk_map = {f.name: f for f in target_fks}
        
        all_fk_names = set(source_fk_map.keys()) | set(target_fk_map.keys())
        fk_diffs = []
        
        for fk_name in sorted(all_fk_names):
            source_fk = source_fk_map.get(fk_name)
            target_fk = target_fk_map.get(fk_name)
            
            if source_fk and target_fk:
                # FK exists in both - check for modifications
                if (source_fk.referenced_table != target_fk.referenced_table or
                    source_fk.referenced_column != target_fk.referenced_column or
                    source_fk.on_delete != target_fk.on_delete or
                    source_fk.on_update != target_fk.on_update):
                    
                    fk_diffs.append(ForeignKeyDiff(
                        fk_name=fk_name,
                        change_type=ChangeType.MODIFIED,
                        old_fk=source_fk,
                        new_fk=target_fk,
                        impact_level=ImpactLevel.MAJOR,
                        impact_description=f"Foreign key '{fk_name}' modified"
                    ))
            elif target_fk and not source_fk:
                # New FK
                fk_diffs.append(ForeignKeyDiff(
                    fk_name=fk_name,
                    change_type=ChangeType.ADDED,
                    new_fk=target_fk,
                    impact_level=ImpactLevel.MAJOR,
                    impact_description=f"New foreign key '{fk_name}' added"
                ))
            else:
                # Removed FK
                fk_diffs.append(ForeignKeyDiff(
                    fk_name=fk_name,
                    change_type=ChangeType.REMOVED,
                    old_fk=source_fk,
                    impact_level=ImpactLevel.MAJOR,
                    impact_description=f"Foreign key '{fk_name}' removed"
                ))
        
        return fk_diffs
    
    def _assess_column_addition_impact(self, column: ColumnInfo, table_name: str) -> Tuple[ImpactLevel, str, str]:
        """Assess the impact of adding a new column."""
        if column.is_primary_key:
            return (ImpactLevel.BREAKING, 
                   f"New primary key column '{column.name}' in table '{table_name}'",
                   "Adding primary key to existing table requires careful planning")
        
        if not column.is_nullable and column.default_value is None:
            return (ImpactLevel.BREAKING,
                   f"New NOT NULL column '{column.name}' without default value",
                   "Must provide default value or make nullable for existing rows")
        
        return (ImpactLevel.MINOR,
               f"New column '{column.name}' added to table '{table_name}'",
               "Safe addition with proper defaults")
    
    def _assess_column_removal_impact(self, column: ColumnInfo, table_name: str) -> Tuple[ImpactLevel, str, str]:
        """Assess the impact of removing a column."""
        if column.is_primary_key:
            return (ImpactLevel.BREAKING,
                   f"Primary key column '{column.name}' removed from table '{table_name}'",
                   "Removing primary key requires careful data migration")
        
        return (ImpactLevel.BREAKING,
               f"Column '{column.name}' removed from table '{table_name}' - potential data loss",
               "Ensure data is backed up or migrated before removal")
    
    def _analyze_dependency_changes(self, table_diffs: List[TableDiff]) -> List[Dict[str, Any]]:
        """Analyze changes in table dependencies."""
        dependency_changes = []
        
        for table_diff in table_diffs:
            if table_diff.foreign_key_diffs:
                for fk_diff in table_diff.foreign_key_diffs:
                    change_info = {
                        "table": table_diff.table_name,
                        "change_type": fk_diff.change_type.value,
                        "foreign_key": fk_diff.fk_name,
                        "impact": fk_diff.impact_level.value
                    }
                    
                    if fk_diff.new_fk:
                        change_info["referenced_table"] = fk_diff.new_fk.referenced_table
                        change_info["referenced_column"] = fk_diff.new_fk.referenced_column
                    elif fk_diff.old_fk:
                        change_info["referenced_table"] = fk_diff.old_fk.referenced_table
                        change_info["referenced_column"] = fk_diff.old_fk.referenced_column
                    
                    dependency_changes.append(change_info)
        
        return dependency_changes
    
    def _assess_overall_impact(self, table_diffs: List[TableDiff]) -> ImpactLevel:
        """Assess the overall impact of all schema changes."""
        max_impact = ImpactLevel.COSMETIC
        
        for table_diff in table_diffs:
            if table_diff.impact_level == ImpactLevel.BREAKING:
                return ImpactLevel.BREAKING
            elif table_diff.impact_level == ImpactLevel.MAJOR and max_impact != ImpactLevel.BREAKING:
                max_impact = ImpactLevel.MAJOR
            elif table_diff.impact_level == ImpactLevel.MINOR and max_impact == ImpactLevel.COSMETIC:
                max_impact = ImpactLevel.MINOR
            
            # Check individual changes
            for col_diff in table_diff.column_diffs:
                if col_diff.impact_level == ImpactLevel.BREAKING:
                    return ImpactLevel.BREAKING
                elif col_diff.impact_level == ImpactLevel.MAJOR and max_impact != ImpactLevel.BREAKING:
                    max_impact = ImpactLevel.MAJOR
                elif col_diff.impact_level == ImpactLevel.MINOR and max_impact == ImpactLevel.COSMETIC:
                    max_impact = ImpactLevel.MINOR
        
        return max_impact
    
    def _generate_migration_strategy(self, table_diffs: List[TableDiff], overall_impact: ImpactLevel) -> str:
        """Generate recommended migration strategy."""
        breaking_changes = sum(1 for td in table_diffs 
                             for cd in td.column_diffs 
                             if cd.impact_level == ImpactLevel.BREAKING)
        
        if overall_impact == ImpactLevel.BREAKING:
            if breaking_changes > 3:
                return "PHASED_MIGRATION"
            else:
                return "CAREFUL_MIGRATION"
        elif overall_impact == ImpactLevel.MAJOR:
            return "STANDARD_MIGRATION"
        else:
            return "SIMPLE_MIGRATION"
    
    def _assess_risks(self, table_diffs: List[TableDiff], overall_impact: ImpactLevel) -> str:
        """Assess migration risks."""
        risks = []
        
        # Check for data loss risks
        data_loss_risk = any(td.change_type == ChangeType.REMOVED for td in table_diffs)
        if data_loss_risk:
            risks.append("HIGH: Potential data loss from table/column removal")
        
        # Check for type conversion risks
        type_changes = sum(1 for td in table_diffs 
                          for cd in td.column_diffs 
                          if cd.change_type == ChangeType.MODIFIED and 
                          cd.old_column and cd.new_column and 
                          cd.old_column.normalized_type != cd.new_column.normalized_type)
        
        if type_changes > 0:
            risks.append(f"MEDIUM: {type_changes} data type conversions required")
        
        # Check for constraint risks
        constraint_changes = sum(1 for td in table_diffs 
                               for cd in td.column_diffs 
                               if cd.impact_level == ImpactLevel.BREAKING)
        
        if constraint_changes > 2:
            risks.append(f"MEDIUM: {constraint_changes} breaking constraint changes")
        
        if overall_impact == ImpactLevel.BREAKING:
            risks.append("HIGH: Breaking changes detected - thorough testing required")
        
        return "; ".join(risks) if risks else "LOW: Minimal migration risks identified"
    
    def _estimate_downtime(self, table_diffs: List[TableDiff], source_schema: DatabaseSchema) -> str:
        """Estimate required downtime for migration."""
        # Get total estimated row counts
        total_rows = sum(table.row_count or 0 for table in source_schema.tables)
        
        # Count complex operations
        complex_operations = sum(1 for td in table_diffs 
                               if td.migration_complexity == "HIGH")
        
        medium_operations = sum(1 for td in table_diffs 
                              if td.migration_complexity == "MEDIUM")
        
        if complex_operations > 0:
            if total_rows > 1000000:  # 1M+ rows
                return "EXTENDED (2-6 hours)"
            elif total_rows > 100000:  # 100K+ rows
                return "SIGNIFICANT (30-120 minutes)"
            else:
                return "MODERATE (10-30 minutes)"
        elif medium_operations > 2:
            return "MODERATE (10-30 minutes)"
        elif any(td.change_type != ChangeType.UNCHANGED for td in table_diffs):
            return "MINIMAL (1-10 minutes)"
        else:
            return "NONE (online migration possible)"
    
    def _assess_rollback_complexity(self, table_diffs: List[TableDiff]) -> str:
        """Assess complexity of rolling back the migration."""
        data_loss_changes = sum(1 for td in table_diffs 
                              if td.change_type == ChangeType.REMOVED or
                              any(cd.change_type == ChangeType.REMOVED for cd in td.column_diffs))
        
        type_changes = sum(1 for td in table_diffs 
                          for cd in td.column_diffs 
                          if cd.change_type == ChangeType.MODIFIED and 
                          cd.old_column and cd.new_column and 
                          cd.old_column.normalized_type != cd.new_column.normalized_type)
        
        if data_loss_changes > 0:
            return "HIGH - Data restoration required"
        elif type_changes > 3:
            return "MEDIUM - Multiple type conversions to reverse"
        else:
            return "LOW - Schema-only changes"
    
    def _initialize_impact_rules(self) -> Dict[str, Any]:
        """Initialize rules for impact assessment."""
        return {
            "breaking_changes": [
                "primary_key_removal",
                "column_removal",
                "table_removal",
                "not_null_addition",
                "type_incompatible_change",
                "length_reduction"
            ],
            "major_changes": [
                "foreign_key_changes",
                "unique_constraint_changes",
                "index_removal",
                "auto_increment_changes"
            ],
            "minor_changes": [
                "column_addition_with_default",
                "index_addition",
                "nullable_to_not_null_with_default",
                "length_increase"
            ]
        }
    
    def generate_html_diff_report(self, schema_diff: SchemaDiff, output_path: str) -> None:
        """Generate an HTML diff report for visual analysis.
        
        Args:
            schema_diff: Schema diff to generate report for
            output_path: Path to save HTML report
        """
        html_content = self._generate_html_report_content(schema_diff)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML diff report saved to {output_path}")
    
    def _generate_html_report_content(self, schema_diff: SchemaDiff) -> str:
        """Generate HTML content for the diff report."""
        # This is a simplified HTML generator - in production you might want to use a template engine
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schema Diff Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .table-diff {{ margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; }}
        .table-header {{ background: #007bff; color: white; padding: 10px; }}
        .change-added {{ background: #d4edda; color: #155724; }}
        .change-removed {{ background: #f8d7da; color: #721c24; }}
        .change-modified {{ background: #fff3cd; color: #856404; }}
        .impact-breaking {{ color: #dc3545; font-weight: bold; }}
        .impact-major {{ color: #fd7e14; font-weight: bold; }}
        .impact-minor {{ color: #28a745; }}
        .details {{ padding: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Database Schema Diff Report</h1>
        <p><strong>Source:</strong> {schema_diff.source_schema.database_name} (v{schema_diff.source_schema.schema_version or 'unknown'})</p>
        <p><strong>Target:</strong> {schema_diff.target_schema.database_name} (v{schema_diff.target_schema.schema_version or 'unknown'})</p>
        <p><strong>Generated:</strong> {schema_diff.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Overall Impact:</strong> <span class="impact-{schema_diff.overall_impact.name.lower()}">{schema_diff.overall_impact.name}</span></p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div><strong>Tables:</strong> {schema_diff.summary.get('total_tables', 0)} total ({schema_diff.summary.get('added_tables', 0)} added, {schema_diff.summary.get('removed_tables', 0)} removed, {schema_diff.summary.get('modified_tables', 0)} modified)</div>
            <div><strong>Columns:</strong> {schema_diff.summary.get('total_columns', 0)} total ({schema_diff.summary.get('added_columns', 0)} added, {schema_diff.summary.get('removed_columns', 0)} removed, {schema_diff.summary.get('modified_columns', 0)} modified)</div>
            <div><strong>Migration Strategy:</strong> {schema_diff.migration_strategy}</div>
            <div><strong>Estimated Downtime:</strong> {schema_diff.estimated_downtime}</div>
            <div><strong>Rollback Complexity:</strong> {schema_diff.rollback_complexity}</div>
            <div><strong>Risk Assessment:</strong> {schema_diff.risk_assessment}</div>
        </div>
    </div>
"""
        
        # Add table diffs
        for table_diff in schema_diff.table_diffs:
            if table_diff.change_type == ChangeType.UNCHANGED:
                continue
                
            change_class = f"change-{table_diff.change_type.name.lower()}"
            impact_class = f"impact-{table_diff.impact_level.name.lower()}"
            
            html += f"""
    <div class="table-diff">
        <div class="table-header {change_class}">
            <h3>Table: {table_diff.table_name} ({table_diff.change_type.name})</h3>
            <span class="{impact_class}">Impact: {table_diff.impact_level.name}</span>
        </div>
        <div class="details">
"""
            
            if table_diff.impact_description:
                html += f"<p><strong>Impact:</strong> {table_diff.impact_description}</p>"
            
            if table_diff.migration_complexity:
                html += f"<p><strong>Migration Complexity:</strong> {table_diff.migration_complexity}</p>"
            
            # Add column changes
            if table_diff.column_diffs:
                html += "<h4>Column Changes</h4><table><tr><th>Column</th><th>Change</th><th>Impact</th><th>Description</th></tr>"
                for col_diff in table_diff.column_diffs:
                    change_class = f"change-{col_diff.change_type.name.lower()}"
                    impact_class = f"impact-{col_diff.impact_level.name.lower()}"
                    html += f"""
                    <tr class="{change_class}">
                        <td>{col_diff.column_name}</td>
                        <td>{col_diff.change_type.name}</td>
                        <td class="{impact_class}">{col_diff.impact_level.name}</td>
                        <td>{col_diff.impact_description}</td>
                    </tr>
"""
                html += "</table>"
            
            # Add index changes
            if table_diff.index_diffs:
                html += "<h4>Index Changes</h4><table><tr><th>Index</th><th>Change</th><th>Impact</th><th>Description</th></tr>"
                for idx_diff in table_diff.index_diffs:
                    change_class = f"change-{idx_diff.change_type.name.lower()}"
                    impact_class = f"impact-{idx_diff.impact_level.name.lower()}"
                    html += f"""
                    <tr class="{change_class}">
                        <td>{idx_diff.index_name}</td>
                        <td>{idx_diff.change_type.name}</td>
                        <td class="{impact_class}">{idx_diff.impact_level.name}</td>
                        <td>{idx_diff.impact_description}</td>
                    </tr>
"""
                html += "</table>"
            
            # Add foreign key changes
            if table_diff.foreign_key_diffs:
                html += "<h4>Foreign Key Changes</h4><table><tr><th>Foreign Key</th><th>Change</th><th>Impact</th><th>Description</th></tr>"
                for fk_diff in table_diff.foreign_key_diffs:
                    change_class = f"change-{fk_diff.change_type.name.lower()}"
                    impact_class = f"impact-{fk_diff.impact_level.name.lower()}"
                    html += f"""
                    <tr class="{change_class}">
                        <td>{fk_diff.fk_name}</td>
                        <td>{fk_diff.change_type.name}</td>
                        <td class="{impact_class}">{fk_diff.impact_level.name}</td>
                        <td>{fk_diff.impact_description}</td>
                    </tr>
"""
                html += "</table>"
            
            html += "</div></div>"
        
        html += """
</body>
</html>
"""
        return html
    
    def save_diff_to_json(self, schema_diff: SchemaDiff, output_path: str) -> None:
        """Save schema diff to JSON file for programmatic analysis.
        
        Args:
            schema_diff: Schema diff to save
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for JSON serialization
        diff_dict = asdict(schema_diff)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(diff_dict, f, indent=2, cls=EnumEncoder)
        
        logger.info(f"Schema diff saved to {output_path}") 