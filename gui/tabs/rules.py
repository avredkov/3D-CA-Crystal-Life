from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from PySide6 import QtWidgets, QtCore

from gui.services.rules_service import (
    index_to_vector,
    vector_to_index,
    load_recipe,
    save_recipe,
    validate_recipe,
    is_protected_target,
    list_rulesets,
    get_recipe_parameters,
)
from gui.components.NeighborhoodPreview import NeighborhoodPreview
from gui.services.rules_service import inverted_self_vector


PROTECTED_RECIPES = {"Default", "Default_with_3D_nucleation"}
NUCLEATION_INDEX = 729 * 1 + 243 * 1 + 81 * 1 + 27 * 1 + 9 * 1 + 3 * 1 + 1


class RulesTableModel(QtCore.QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._values: List[float] = [0.0] * (3 ** 7)
        self._expr: List[str] = ["0"] * (3 ** 7)  # symbolic selection
        # Mapping from parameter key -> human label for display in Value column
        self._label_by_key: dict[str, str] = {}

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 3 ** 7

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        # index, vector, value
        return 3

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or orientation != QtCore.Qt.Horizontal:
            return None
        return ["Index", "Vector [Self, Right, Down, Left, Up, Back, Front]", "Value"][section]

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return row
            if col == 1:
                return str(index_to_vector(row))
            if col == 2:
                sym = (self._expr[row] or "").strip()
                # When symbol is "0" or "1" or empty -> show numeric value
                if sym in ("", "0", "1"):
                    try:
                        return f"{float(self._values[row]):.6g}"
                    except Exception:
                        return "0"
                # Otherwise, show human label if available, falling back to key
                return self._label_by_key.get(sym, sym)
        return None

    def setData(self, index: QtCore.QModelIndex, value, role: int = QtCore.Qt.EditRole):
        if not index.isValid() or index.column() != 2:
            return False
        row = index.row()
        # Accept symbolic recipe keys or numeric constants. No 'custom' string.
        if isinstance(value, str):
            symbol = value.strip()
            if symbol in ("0", "1"):
                self._expr[row] = symbol
                self._values[row] = 0.0 if symbol == "0" else 1.0
            else:
                # Treat any other string as a symbolic parameter key from recipe
                self._expr[row] = symbol
                # Keep last numeric; resolution to actual value happens at assign time
                try:
                    v = float(self._values[row])
                except Exception:
                    v = 0.0
                self._values[row] = max(0.0, min(1.0, v))
        else:
            try:
                v = float(value)
            except Exception:
                return False
            v = max(0.0, min(1.0, v))  # probabilities always clamped 0..1
            self._expr[row] = ""  # numeric literal, no symbol
            self._values[row] = v
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if index.column() == 2:
            flags |= QtCore.Qt.ItemIsEditable
        return flags

    def toggle_nucleation(self, on: bool) -> None:
        self._expr[NUCLEATION_INDEX] = "1" if on else "0"
        self._values[NUCLEATION_INDEX] = 1.0 if on else 0.0
        idx = self.index(NUCLEATION_INDEX, 2)
        self.dataChanged.emit(idx, idx)

    def set_dataset(self, expr: List[str], values: List[float]) -> None:
        if len(values) != 3 ** 7 or len(expr) != 3 ** 7:
            raise ValueError("invalid dataset lengths")
        self.beginResetModel()
        self._expr = list(expr)
        self._values = list(values)
        self.endResetModel()

    def get_dataset(self) -> tuple[List[str], List[float]]:
        return self._expr, self._values

    def set_label_mapping(self, mapping: dict[str, str]) -> None:
        """Set key->label mapping used for displaying symbolic values as labels."""
        self._label_by_key = dict(mapping or {})
        try:
            top_left = self.index(0, 2)
            bottom_right = self.index(self.rowCount() - 1, 2)
            self.dataChanged.emit(top_left, bottom_right)
        except Exception:
            pass


class RulesTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        # Track original parameter keys to support rename propagation
        self._orig_param_keys: list[str] = []

        # Top controls wrapped in group box (similar to Initialization section)
        select_group = QtWidgets.QGroupBox("Select recipe")
        ctrl = QtWidgets.QHBoxLayout(select_group)
        self.recipe_name = QtWidgets.QComboBox(); self.recipe_name.setEditable(False)
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save_as = QtWidgets.QPushButton("Save As…")
        self.btn_delete = QtWidgets.QPushButton("Delete recipe")
        tip = (
            "Built-in recipes ('Default', 'Default_with_3D_nucleation') cannot be saved or deleted.\n"
            "Use 'Save As…' to create a custom recipe, then 'Save' and 'Delete' operate on custom recipes."
        )
        self.btn_save.setToolTip(tip)
        self.btn_save_as.setToolTip(tip)
        self.btn_delete.setToolTip(tip)
        self.chk_nucl = QtWidgets.QCheckBox("Enable 3D nucleation")
        for w in (self.recipe_name, self.btn_save, self.btn_save_as, self.btn_delete, self.chk_nucl):
            ctrl.addWidget(w)
        root.addWidget(select_group)

        # Subtabs: Rules and Recipe parameters
        self.subtabs = QtWidgets.QTabWidget()
        # Rules subtab (existing table)
        page_rules = QtWidgets.QWidget(); page_rules_l = QtWidgets.QVBoxLayout(page_rules)
        self.model = RulesTableModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(False)
        # Install delegate/editor for Value column
        class ValueDelegate(QtWidgets.QStyledItemDelegate):
            def createEditor(self, parent, option, index):
                if index.column() != 2:
                    return super().createEditor(parent, option, index)
                editor = QtWidgets.QComboBox(parent)
                editor.setEditable(True)
                # Build choices: 0, 1, and recipe parameter labels (store keys in item data)
                try:
                    name = (self.parent().recipe_name.currentText() or "Default").strip()
                    params = get_recipe_parameters(name) or []
                    # Prepare mapping label -> key (fallback label to key if missing)
                    label_key_pairs = []
                    for p in params:
                        key = str(p.get("key") or "")
                        if not key:
                            continue
                        label = str(p.get("label") or key)
                        label_key_pairs.append((label, key))
                except Exception:
                    label_key_pairs = []
                editor.addItem("0")
                editor.addItem("1")
                # Add labels and keep original keys in UserRole for reverse mapping
                for label, key in label_key_pairs:
                    editor.addItem(label)
                    try:
                        idx = editor.findText(label)
                        if idx >= 0:
                            editor.setItemData(idx, key, QtCore.Qt.UserRole)
                    except Exception:
                        pass
                # Ensure free numeric entry allowed
                editor.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
                return editor
            def setEditorData(self, editor, index):
                if isinstance(editor, QtWidgets.QComboBox):
                    current = index.model().data(index, QtCore.Qt.DisplayRole) or ""
                    current_str = str(current)
                    # Direct matches for 0/1
                    pos = editor.findText(current_str)
                    if pos >= 0:
                        editor.setCurrentIndex(pos)
                        return
                    # If symbolic key, find item whose UserRole matches key and select its label
                    try:
                        for i in range(editor.count()):
                            key_data = editor.itemData(i, QtCore.Qt.UserRole)
                            if key_data and str(key_data) == current_str:
                                editor.setCurrentIndex(i)
                                return
                    except Exception:
                        pass
                    # Otherwise, treat as numeric/string input
                    editor.setEditText(current_str)
                else:
                    super().setEditorData(editor, index)
            def setModelData(self, editor, model, index):
                if isinstance(editor, QtWidgets.QComboBox):
                    text = (editor.currentText() or "").strip()
                    # If the selected item corresponds to a label, retrieve its key (UserRole)
                    try:
                        sel_idx = editor.currentIndex()
                        if sel_idx >= 0:
                            key_data = editor.itemData(sel_idx, QtCore.Qt.UserRole)
                            if key_data:
                                model.setData(index, str(key_data))
                                return
                    except Exception:
                        pass
                    # Otherwise, allow only numeric input in [0,1], including textual '0'/'1'. Reject other strings.
                    try:
                        val = float(text)
                        if val < 0.0 or val > 1.0:
                            raise ValueError("out of range")
                        model.setData(index, val)
                        return
                    except Exception:
                        # Invalid manual text (e.g., parameter name typed). Reject and keep previous value.
                        try:
                            QtWidgets.QMessageBox.warning(self.parent(), "Invalid input", "Type a number in [0,1], or choose a parameter from the dropdown.")
                        except Exception:
                            pass
                        return
                else:
                    super().setModelData(editor, model, index)
        self.table.setItemDelegateForColumn(2, ValueDelegate(self))
        try:
            self.table.setEditTriggers(
                QtWidgets.QAbstractItemView.DoubleClicked
                | QtWidgets.QAbstractItemView.EditKeyPressed
                | QtWidgets.QAbstractItemView.SelectedClicked
            )
        except Exception:
            pass
        self.table.horizontalHeader().setStretchLastSection(True)
        try:
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
            hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
            self.table.setColumnWidth(2, 120)
        except Exception:
            pass
        page_rules_l.addWidget(self.table)
        # Hint below the rules table
        hint = QtWidgets.QLabel(
            "If Self is a mobile atom (1), the probability in Value column refers to growth (transition to crystalline state 2).\n"
            "If Self is a crystalline atom (2), the probability in Value column refers to sublimation (transition to mobile state 1)."
        )
        hint.setWordWrap(True)
        page_rules_l.addWidget(hint)

        # Illustration row: left preview + caption, center label, right preview + caption
        illustr_row = QtWidgets.QHBoxLayout()
        self.preview_left = NeighborhoodPreview(self)
        self.preview_right = NeighborhoodPreview(self)
        # Ensure previews expand horizontally and have reduced width to favor text space
        try:
            for p in (self.preview_left, self.preview_right):
                p.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
                # Narrower previews (width) to give more room for text in the middle
                p.setMinimumWidth(160)
                try:
                    p.setMaximumWidth(260)
                except Exception:
                    pass
                # Reduce vertical footprint by ~2x for the previews
                if hasattr(p, "view"):
                    try:
                        p.view.setMinimumHeight(130)
                        p.view.setMaximumHeight(200)
                    except Exception:
                        pass
            # Make sure right preview is visible in case it was hidden elsewhere
            self.preview_right.setVisible(True)
        except Exception:
            pass
        # Probability label between previews
        self.prob_label = QtWidgets.QLabel("…translates with probability <prob> into…")
        self.prob_label.setAlignment(QtCore.Qt.AlignCenter)
        # Keep on a single line; allow it to expand horizontally
        try:
            self.prob_label.setWordWrap(False)
            self.prob_label.setTextFormat(QtCore.Qt.RichText)
            self.prob_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass

        # Show legend outside previews to avoid clipping; hide internal legends
        try:
            if hasattr(self.preview_left, "legend"):
                self.preview_left.legend.setVisible(False)
            if hasattr(self.preview_right, "legend"):
                self.preview_right.legend.setVisible(False)
        except Exception:
            pass

        # Show Reset view button only on left preview, but make it reset both previews
        try:
            if hasattr(self.preview_right, "btn_reset"):
                self.preview_right.btn_reset.setVisible(False)
            if hasattr(self.preview_left, "btn_reset"):
                # Ensure single connection
                try:
                    self.preview_left.btn_reset.clicked.disconnect()
                except Exception:
                    pass
                self.preview_left.btn_reset.clicked.connect(
                    lambda: (self.preview_left.reset_view(), self.preview_right.reset_view())
                )
        except Exception:
            pass

        # Wrap previews with captions below them
        left_col = QtWidgets.QVBoxLayout(); left_caption = QtWidgets.QLabel("Current state")
        try:
            left_caption.setAlignment(QtCore.Qt.AlignCenter)
            left_caption.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass
        left_col.addWidget(self.preview_left)
        left_col.addWidget(left_caption)
        left_container = QtWidgets.QWidget(); left_container.setLayout(left_col)
        try:
            left_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass

        right_col = QtWidgets.QVBoxLayout(); right_caption = QtWidgets.QLabel("Next state")
        try:
            right_caption.setAlignment(QtCore.Qt.AlignCenter)
            right_caption.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass
        right_col.addWidget(self.preview_right)
        right_col.addWidget(right_caption)
        right_container = QtWidgets.QWidget(); right_container.setLayout(right_col)
        try:
            right_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass

        # Make label wider by adjusting stretch: left 1, label 3, right 1 (containers equal sizes)
        illustr_row.addWidget(left_container, 1)
        illustr_row.addWidget(self.prob_label, 3)
        illustr_row.addWidget(right_container, 1)

        # Keep label centered; previews expand equally via stretch factors
        illustr_row.setAlignment(self.prob_label, QtCore.Qt.AlignCenter)

        page_rules_l.addLayout(illustr_row)
        # Bias layout to give more vertical space to the table above
        try:
            page_rules_l.setStretch(page_rules_l.indexOf(self.table), 3)
        except Exception:
            pass

        # Global legend label (always visible, wraps if needed)
        try:
            self.legend_label = QtWidgets.QLabel(
                "0: empty  1: mobile (yellow)  2: crystalline (green)    Dirs: Self, +X Right, +Y Down, -X Left, -Y Up, -Z Back, +Z Front"
            )
            self.legend_label.setWordWrap(True)
            self.legend_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            page_rules_l.addWidget(self.legend_label)
        except Exception:
            pass
        self.subtabs.addTab(page_rules, "Rules")

        # Recipe parameters subtab (read-only for built-ins in this phase)
        page_params = QtWidgets.QWidget(); page_params_l = QtWidgets.QVBoxLayout(page_params)
        
        self.params_table = QtWidgets.QTableWidget(0, 4)
        self.params_table.setHorizontalHeaderLabels(["Key", "Label", "Description", "Default value"])
        try:
            hh = self.params_table.horizontalHeader()
            # Narrow columns for Key, Label, Default value; stretch Description
            hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
            hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            self.params_table.setWordWrap(True)
        except Exception:
            pass
        # Parameter table delegate: spinbox 0..1 for Default column, line edits otherwise
        class ParamDelegate(QtWidgets.QStyledItemDelegate):
            def createEditor(self, parent, option, index):
                if index.column() == 3:
                    spin = QtWidgets.QDoubleSpinBox(parent)
                    spin.setRange(0.0, 1.0)
                    spin.setSingleStep(0.01)
                    return spin
                return QtWidgets.QLineEdit(parent)
            def setEditorData(self, editor, index):
                val = index.model().data(index, QtCore.Qt.DisplayRole) or ""
                if isinstance(editor, QtWidgets.QDoubleSpinBox):
                    try:
                        editor.setValue(float(val))
                    except Exception:
                        editor.setValue(0.0)
                else:
                    editor.setText(str(val))
            def setModelData(self, editor, model, index):
                if isinstance(editor, QtWidgets.QDoubleSpinBox):
                    v = max(0.0, min(1.0, float(editor.value())))
                    model.setData(index, f"{v:.6g}")
                else:
                    model.setData(index, editor.text())
        self.params_table.setItemDelegate(ParamDelegate(self.params_table))
        try:
            # Keep last column sizing policy even if style changes
            self.params_table.horizontalHeader().setStretchLastSection(False)
        except Exception:
            pass
        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_param = QtWidgets.QPushButton("Add parameter")
        self.btn_del_param = QtWidgets.QPushButton("Delete parameter")
        btn_row.addWidget(self.btn_add_param)
        btn_row.addWidget(self.btn_del_param)
        btn_row.addStretch(1)
        page_params_l.addLayout(btn_row)
        page_params_l.addWidget(self.params_table)
        self.subtabs.addTab(page_params, "Recipe parameters")
        root.addWidget(self.subtabs)

        # wiring
        self.btn_save.clicked.connect(self._on_save_recipe)
        self.btn_save_as.clicked.connect(self._on_save_as)
        self.btn_delete.clicked.connect(self._on_delete_recipe)
        self.chk_nucl.toggled.connect(self._on_toggle_nucleation)
        self.table.selectionModel().selectionChanged.connect(self._on_row_change)
        # Update probability label on edits to Value column
        self.model.dataChanged.connect(self._on_model_changed)
        # Non-editable dropdown: guard on selection change
        self.recipe_name.currentTextChanged.connect(self._on_select_change)
        # Populate parameters subtab when recipe changes
        self.recipe_name.currentTextChanged.connect(self._populate_params_subtab)
        # Wire parameter buttons
        self.btn_add_param.clicked.connect(self._on_add_param)
        self.btn_del_param.clicked.connect(self._on_delete_param)
        # populate on load
        try:
            self._refresh_rulesets()
            # default selection
            idx = self.recipe_name.findText("Default")
            if idx >= 0:
                self.recipe_name.setCurrentIndex(idx)
            else:
                self.recipe_name.setEditText("Default")
            # load default recipe
            self._load_into_model("Default")
        except Exception:
            pass

        # default selection and guard
        self._guard_buttons()
        # Initialize previews and label to first row
        try:
            self.table.selectRow(0)
            self._refresh_previews_and_label(0)
        except Exception:
            pass
        # Populate params subtab initially
        try:
            self._populate_params_subtab()
        except Exception:
            pass
        # Initialize label mapping for display based on current recipe
        try:
            self._refresh_value_label_mapping()
        except Exception:
            pass

    def _guard_buttons(self) -> None:
        name = (self.recipe_name.currentText() or "").strip()
        is_protected = name in PROTECTED_RECIPES or is_protected_target(name)
        self.btn_save.setEnabled(not is_protected)
        self.btn_delete.setEnabled(bool(name and not is_protected))

    def _on_row_change(self) -> None:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return
        row = idxs[0].row()
        self._refresh_previews_and_label(row)

    def _on_model_changed(self, topLeft: QtCore.QModelIndex, bottomRight: QtCore.QModelIndex) -> None:
        # If Value column of selected row changed, refresh label
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return
        row = idxs[0].row()
        # Refresh regardless of column range to keep simple and robust
        self._refresh_previews_and_label(row)

    def _refresh_previews_and_label(self, row: int) -> None:
        vec = index_to_vector(row)
        # Left preview: vector as-is
        self.preview_left.set_vector(vec)
        # Probability / symbol at this row
        expr, vals = self.model._expr, self.model._values  # accessing internal lists intentionally
        symbol = (expr[row] or "").strip()
        v = 0.0
        try:
            v = float(vals[row])
        except Exception:
            v = 0.0
        self_state = int(vec[0]) if len(vec) >= 1 else 0
        # Resolve human-friendly label for symbolic parameters
        is_symbolic = symbol not in ("", "0", "1")
        display_label = symbol
        if is_symbolic:
            try:
                name = (self.recipe_name.currentText() or "Default").strip()
                params = get_recipe_parameters(name) or []
                for p in params:
                    if str(p.get("key")) == symbol:
                        display_label = str(p.get("label") or symbol)
                        break
            except Exception:
                pass
        # Cases:
        # 1) Self == 0 -> remains the same; right preview mirrors left
        # 2) Symbolic expression used -> treat as transition using expression label (invert preview)
        # 3) prob == 0 (symbol "0" or numeric ~0) and Self != 0 -> remains the same; right mirrors left
        # 4) Otherwise -> invert Self and show bold numeric probability
        if self_state == 0:
            self.preview_right.set_vector(vec)
            self.prob_label.setText("…remains the same as the Self position is empty")
        elif is_symbolic:
            inv_vec = inverted_self_vector(vec)
            self.preview_right.set_vector(inv_vec)
            self.prob_label.setText(f"…translates with probability <b>{display_label}</b> into…")
        elif symbol == "0" or abs(v) < 1e-12:
            self.preview_right.set_vector(vec)
            self.prob_label.setText("…remains the same as the transition probability is <b>0</b>")
        else:
            inv_vec = inverted_self_vector(vec)
            self.preview_right.set_vector(inv_vec)
            self.prob_label.setText(f"…translates with probability <b>{v:.6g}</b> into…")

    def _on_select_change(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        self._guard_buttons()
        self._load_into_model(name)
        # refresh params table
        try:
            self._populate_params_subtab()
        except Exception:
            pass
        # refresh label mapping for display of Value column
        try:
            self._refresh_value_label_mapping()
        except Exception:
            pass

    def _on_save_recipe(self) -> None:
        name = (self.recipe_name.currentText() or "").strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Save", "Enter a recipe name first")
            return
        if name in PROTECTED_RECIPES:
            QtWidgets.QMessageBox.warning(self, "Protected", "Cannot overwrite a protected recipe. Use Save As…")
            return
        try:
            expr, vals = self.model.get_dataset()
            params = self._collect_params_from_table()
            # validate default range
            for p in params:
                dv = float(p.get("default", 0.0))
                if not (0.0 <= dv <= 1.0):
                    QtWidgets.QMessageBox.warning(self, "Invalid default", f"Default for {p.get('key')} must be in [0,1]")
                    return
            # propagate parameter key renames into RULE_EXPR (expr)
            try:
                renames: dict[str, str] = {}
                n = min(len(self._orig_param_keys), len(params))
                for i in range(n):
                    old = (self._orig_param_keys[i] or "").strip()
                    new = (params[i].get("key") or "").strip()
                    if old and new and old != new:
                        renames[old] = new
                if renames:
                    expr = [renames.get((e or "").strip(), (e or "")) for e in expr]
            except Exception:
                pass
            saved = save_recipe(name, expr, vals, overwrite=True, parameters=params)
            # Validate import
            try:
                validate_recipe(name)
                QtWidgets.QMessageBox.information(self, "Saved", f"Recipe written and validated: {saved}")
            except Exception as ve:
                QtWidgets.QMessageBox.warning(self, "Validation failed", f"Saved, but validation error: {ve}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _on_save_as(self) -> None:
        # Always prompt for a new name; typing in dropdown alone does not save
        name, ok = QtWidgets.QInputDialog.getText(self, "Save recipe as", "Enter new recipe name:")
        if not ok or not name:
            return
        if name in PROTECTED_RECIPES:
            QtWidgets.QMessageBox.warning(self, "Protected name", "Choose a different name (protected)")
            return
        try:
            expr, vals = self.model.get_dataset()
            # If saving under new name, use parameters from the editable table
            params = self._collect_params_from_table()
            for p in params:
                dv = float(p.get("default", 0.0))
                if not (0.0 <= dv <= 1.0):
                    QtWidgets.QMessageBox.warning(self, "Invalid default", f"Default for {p.get('key')} must be in [0,1]")
                    return
            # propagate renames based on original keys snapshot
            try:
                renames: dict[str, str] = {}
                n = min(len(self._orig_param_keys), len(params))
                for i in range(n):
                    old = (self._orig_param_keys[i] or "").strip()
                    new = (params[i].get("key") or "").strip()
                    if old and new and old != new:
                        renames[old] = new
                if renames:
                    expr = [renames.get((e or "").strip(), (e or "")) for e in expr]
            except Exception:
                pass
            saved = save_recipe(name, expr, vals, overwrite=False, parameters=params)
            # refresh dropdown to include the newly saved recipe
            self._refresh_rulesets()
            self.recipe_name.setEditText(name)
            self._guard_buttons()
            # Validate import
            try:
                validate_recipe(name)
                QtWidgets.QMessageBox.information(self, "Saved", f"Recipe written and validated: {saved}")
            except Exception as ve:
                QtWidgets.QMessageBox.warning(self, "Validation failed", f"Saved, but validation error: {ve}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _on_delete_recipe(self) -> None:
        name = (self.recipe_name.currentText() or "").strip()
        if not name or name in PROTECTED_RECIPES:
            return
        resp = QtWidgets.QMessageBox.question(self, "Delete recipe", f"Delete ruleset '{name}' from rules/?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if resp != QtWidgets.QMessageBox.Yes:
            return
        try:
            p = Path("rules") / f"{name}.py"
            if p.exists():
                p.unlink()
            # refresh and switch back to Default
            self._refresh_rulesets()
            if self.recipe_name.findText("Default") >= 0:
                self.recipe_name.setCurrentText("Default")
                self._load_into_model("Default")
            else:
                # nothing else present
                self.model.set_dataset(["0"] * (3 ** 7), [0.0] * (3 ** 7))
            self._guard_buttons()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Delete failed", str(e))

    def _refresh_rulesets(self) -> None:
        names = list_rulesets() or []
        if not names:
            names = ["Default"]
        self.recipe_name.clear()
        self.recipe_name.addItems(names)
        self.recipe_name.setEditable(False)

    def _load_into_model(self, name: str) -> None:
        try:
            expr, vals = load_recipe(name)
            self.model.set_dataset(expr, vals)
            self.chk_nucl.setChecked(vals[NUCLEATION_INDEX] >= 0.5)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))

    def _populate_params_subtab(self) -> None:
        name = (self.recipe_name.currentText() or "Default").strip()
        try:
            params = get_recipe_parameters(name) or []
        except Exception:
            params = []
        self.params_table.setRowCount(0)
        self._orig_param_keys = []
        for p in params:
            row = self.params_table.rowCount()
            self.params_table.insertRow(row)
            self.params_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(p.get("key", ""))))
            self.params_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(p.get("label", ""))))
            self.params_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(p.get("description", ""))))
            self.params_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(p.get('default', 0.0)):.6g}"))
            self._orig_param_keys.append(str(p.get("key", "")))
        try:
            self.params_table.resizeRowsToContents()
        except Exception:
            pass
        # Update label mapping as parameters may have changed for this recipe
        try:
            self._refresh_value_label_mapping()
        except Exception:
            pass

    def _refresh_value_label_mapping(self) -> None:
        """Build key->label mapping for current recipe and provide it to the table model."""
        name = (self.recipe_name.currentText() or "Default").strip()
        mapping: dict[str, str] = {}
        try:
            params = get_recipe_parameters(name) or []
            for p in params:
                key = str(p.get("key") or "")
                if not key:
                    continue
                label = str(p.get("label") or key)
                mapping[key] = label
        except Exception:
            mapping = {}
        self.model.set_label_mapping(mapping)

    def _collect_params_from_table(self) -> list[dict]:
        params: list[dict] = []
        for row in range(self.params_table.rowCount()):
            key = (self.params_table.item(row, 0).text() if self.params_table.item(row, 0) else "").strip()
            label = (self.params_table.item(row, 1).text() if self.params_table.item(row, 1) else "").strip()
            desc = (self.params_table.item(row, 2).text() if self.params_table.item(row, 2) else "").strip()
            try:
                default = float(self.params_table.item(row, 3).text()) if self.params_table.item(row, 3) else 0.0
            except Exception:
                default = 0.0
            default = max(0.0, min(1.0, default))
            params.append({
                "key": key,
                "label": label,
                "description": desc,
                "default": default,
            })
        return params

    def _on_add_param(self) -> None:
        # Append a new empty row with a unique placeholder key
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)
        self.params_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"param_{row}"))
        self.params_table.setItem(row, 1, QtWidgets.QTableWidgetItem("Label"))
        self.params_table.setItem(row, 2, QtWidgets.QTableWidgetItem("Description"))
        self.params_table.setItem(row, 3, QtWidgets.QTableWidgetItem("0.0"))

    def _on_delete_param(self) -> None:
        rows = sorted({idx.row() for idx in self.params_table.selectedIndexes()})
        if not rows:
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete parameter",
            "Are you sure to delete parameter? All the rules using this parameter will be automatically changed to zero value.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        # Collect keys to remove
        keys_to_remove: set[str] = set()
        for r in rows:
            it = self.params_table.item(r, 0)
            if it:
                keys_to_remove.add((it.text() or "").strip())
        # Remove rows bottom-up
        for r in reversed(rows):
            self.params_table.removeRow(r)
        # Reflect removal into the current rules dataset: replace matching symbols with "0"
        if keys_to_remove:
            try:
                expr, vals = self.model.get_dataset()
                changed = False
                for i, sym in enumerate(list(expr)):
                    if (sym or "").strip() in keys_to_remove:
                        expr[i] = "0"
                        vals[i] = 0.0
                        changed = True
                if changed:
                    self.model.set_dataset(expr, vals)
            except Exception:
                pass

    def _on_toggle_nucleation(self, on: bool) -> None:
        try:
            self.model.toggle_nucleation(on)
            # Focus nucleation rule row and center it
            self.table.selectRow(NUCLEATION_INDEX)
            self.table.scrollTo(self.model.index(NUCLEATION_INDEX, 0), QtWidgets.QAbstractItemView.PositionAtCenter)
            # Update previews/label to reflect that row
            self._refresh_previews_and_label(NUCLEATION_INDEX)
        except Exception:
            pass


