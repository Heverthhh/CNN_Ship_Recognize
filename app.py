# ============================================================
# app.py
# INTERFAZ FINAL - CLASIFICADOR BARCO / NO BARCO
# EfficientNet-B0
# ============================================================

from pathlib import Path
import csv
import queue
import threading
import time

import customtkinter as ctk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# IMPORTAR MODELO FINAL
# ============================================================

from src.predict import (
    predict_image,
    get_model_info,
    load_model,
)


# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# EXTENSIONES PERMITIDAS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# APP
# ============================================================

class ShipClassifierApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        # ====================================================
        # VENTANA
        # ====================================================

        self.title(
            "Clasificador de Barcos - EfficientNet-B0"
        )

        self.geometry(
            "1500x900"
        )

        self.minsize(
            1250,
            760
        )


        # ====================================================
        # VARIABLES
        # ====================================================

        self.image_paths = []

        self.current_index = 0

        self.results = {}

        self.csv_labels = {}

        self.manual_labels = {}

        self.csv_path = None

        self.current_folder = None


        # ====================================================
        # PROCESAMIENTO
        # ====================================================

        self.processing = False

        self.stop_event = threading.Event()

        self.worker_queue = queue.Queue()

        self.processing_thread = None


        # ====================================================
        # TTA
        # ====================================================

        self.tta_var = ctk.BooleanVar(
            value=False
        )


        # ====================================================
        # MODELO
        # ====================================================

        self.model_info = {}

        self.model_ready = False


        # ====================================================
        # GRID
        # ====================================================

        self.grid_columnconfigure(
            0,
            weight=0
        )

        self.grid_columnconfigure(
            1,
            weight=1
        )

        self.grid_columnconfigure(
            2,
            weight=0
        )

        self.grid_rowconfigure(
            0,
            weight=1
        )


        # ====================================================
        # UI
        # ====================================================

        self.create_sidebar()

        self.create_center_panel()

        self.create_right_panel()


        # ====================================================
        # ATAJOS
        # ====================================================

        self.bind(
            "<Left>",
            lambda event:
                self.previous_image()
        )

        self.bind(
            "<Right>",
            lambda event:
                self.next_image()
        )

        self.bind(
            "<space>",
            lambda event:
                self.predict_current()
        )


        # ====================================================
        # CARGAR MODELO
        # ====================================================

        self.after(
            200,
            self.initialize_model
        )


        # ====================================================
        # POLLING DE THREAD
        # ====================================================

        self.after(
            100,
            self.poll_worker_queue
        )


    # ========================================================
    # PANEL IZQUIERDO
    # ========================================================

    def create_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self,
            width=280,
            corner_radius=0
        )

        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.sidebar.grid_propagate(
            False
        )


        # ----------------------------------------------------
        # TÍTULO
        # ----------------------------------------------------

        title = ctk.CTkLabel(
            self.sidebar,
            text="SHIP AI",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            )
        )

        title.pack(
            pady=(
                25,
                0
            )
        )


        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Clasificador Binario",
            font=ctk.CTkFont(
                size=15
            )
        )

        subtitle.pack(
            pady=(
                0,
                18
            )
        )


        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        self.model_status = ctk.CTkLabel(
            self.sidebar,
            text="Cargando modelo...",
            justify="left",
            anchor="w"
        )

        self.model_status.pack(
            fill="x",
            padx=18,
            pady=10
        )


        self.separator_1 = ctk.CTkFrame(
            self.sidebar,
            height=2
        )

        self.separator_1.pack(
            fill="x",
            padx=15,
            pady=10
        )


        # ----------------------------------------------------
        # CARGAR CARPETA
        # ----------------------------------------------------

        self.btn_folder = ctk.CTkButton(
            self.sidebar,
            text="📁  Cargar carpeta",
            height=42,
            command=self.select_folder
        )

        self.btn_folder.pack(
            fill="x",
            padx=18,
            pady=6
        )


        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        self.btn_csv = ctk.CTkButton(
            self.sidebar,
            text="📄  Cargar labels.csv",
            height=42,
            command=self.select_csv
        )

        self.btn_csv.pack(
            fill="x",
            padx=18,
            pady=6
        )


        self.labels_status = ctk.CTkLabel(
            self.sidebar,
            text="Etiquetas: no cargadas",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(
                size=12
            )
        )

        self.labels_status.pack(
            fill="x",
            padx=18,
            pady=(
                2,
                12
            )
        )


        # ----------------------------------------------------
        # TTA
        # ----------------------------------------------------

        self.tta_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Test-Time Augmentation",
            variable=self.tta_var,
            onvalue=True,
            offvalue=False
        )

        self.tta_switch.pack(
            padx=18,
            pady=12,
            anchor="w"
        )


        tta_info = ctk.CTkLabel(
            self.sidebar,
            text=(
                "OFF = mayor velocidad\n"
                "ON = 6 vistas por imagen"
            ),
            justify="left",
            font=ctk.CTkFont(
                size=11
            )
        )

        tta_info.pack(
            padx=18,
            anchor="w"
        )


        self.separator_2 = ctk.CTkFrame(
            self.sidebar,
            height=2
        )

        self.separator_2.pack(
            fill="x",
            padx=15,
            pady=15
        )


        # ----------------------------------------------------
        # PROCESAR ACTUAL
        # ----------------------------------------------------

        self.btn_predict = ctk.CTkButton(
            self.sidebar,
            text="🔍  Clasificar imagen",
            height=44,
            command=self.predict_current
        )

        self.btn_predict.pack(
            fill="x",
            padx=18,
            pady=6
        )


        # ----------------------------------------------------
        # PROCESAR TODO
        # ----------------------------------------------------

        self.btn_process_all = ctk.CTkButton(
            self.sidebar,
            text="▶  Procesar carpeta",
            height=44,
            command=self.start_folder_processing
        )

        self.btn_process_all.pack(
            fill="x",
            padx=18,
            pady=6
        )


        # ----------------------------------------------------
        # DETENER
        # ----------------------------------------------------

        self.btn_stop = ctk.CTkButton(
            self.sidebar,
            text="■  Detener",
            height=40,
            command=self.stop_processing,
            state="disabled"
        )

        self.btn_stop.pack(
            fill="x",
            padx=18,
            pady=6
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        self.progress = ctk.CTkProgressBar(
            self.sidebar
        )

        self.progress.pack(
            fill="x",
            padx=18,
            pady=(
                15,
                5
            )
        )

        self.progress.set(
            0
        )


        self.progress_label = ctk.CTkLabel(
            self.sidebar,
            text="0 / 0",
            font=ctk.CTkFont(
                size=12
            )
        )

        self.progress_label.pack()


        # ----------------------------------------------------
        # MATRIX
        # ----------------------------------------------------

        self.btn_matrix = ctk.CTkButton(
            self.sidebar,
            text="▦  Matriz de confusión",
            command=self.show_confusion_matrix
        )

        self.btn_matrix.pack(
            fill="x",
            padx=18,
            pady=(
                18,
                6
            )
        )


        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        self.btn_export = ctk.CTkButton(
            self.sidebar,
            text="💾  Exportar resultados",
            command=self.export_results
        )

        self.btn_export.pack(
            fill="x",
            padx=18,
            pady=6
        )


    # ========================================================
    # PANEL CENTRAL
    # ========================================================

    def create_center_panel(self):

        self.center_panel = ctk.CTkFrame(
            self,
            corner_radius=12
        )

        self.center_panel.grid(
            row=0,
            column=1,
            padx=12,
            pady=12,
            sticky="nsew"
        )

        self.center_panel.grid_columnconfigure(
            0,
            weight=1
        )

        self.center_panel.grid_rowconfigure(
            1,
            weight=1
        )


        # ----------------------------------------------------
        # NOMBRE IMAGEN
        # ----------------------------------------------------

        self.filename_label = ctk.CTkLabel(
            self.center_panel,
            text="Carga una carpeta de imágenes",
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            )
        )

        self.filename_label.grid(
            row=0,
            column=0,
            padx=15,
            pady=(
                15,
                5
            )
        )


        # ----------------------------------------------------
        # IMAGEN
        # ----------------------------------------------------

        self.image_label = ctk.CTkLabel(
            self.center_panel,
            text="SIN IMAGEN",
            width=570,
            height=570,
            corner_radius=10
        )

        self.image_label.grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="nsew"
        )


        # ----------------------------------------------------
        # NAVEGACIÓN
        # ----------------------------------------------------

        navigation = ctk.CTkFrame(
            self.center_panel,
            fg_color="transparent"
        )

        navigation.grid(
            row=2,
            column=0,
            pady=10
        )


        self.btn_previous = ctk.CTkButton(
            navigation,
            text="← Anterior",
            width=130,
            command=self.previous_image
        )

        self.btn_previous.pack(
            side="left",
            padx=10
        )


        self.position_label = ctk.CTkLabel(
            navigation,
            text="0 / 0",
            width=100
        )

        self.position_label.pack(
            side="left",
            padx=10
        )


        self.btn_next = ctk.CTkButton(
            navigation,
            text="Siguiente →",
            width=130,
            command=self.next_image
        )

        self.btn_next.pack(
            side="left",
            padx=10
        )


        # ----------------------------------------------------
        # ETIQUETA REAL
        # ----------------------------------------------------

        label_frame = ctk.CTkFrame(
            self.center_panel
        )

        label_frame.grid(
            row=3,
            column=0,
            padx=20,
            pady=(
                8,
                15
            ),
            sticky="ew"
        )


        label_frame.grid_columnconfigure(
            1,
            weight=1
        )


        ctk.CTkLabel(
            label_frame,
            text="Etiqueta real:"
        ).grid(
            row=0,
            column=0,
            padx=10,
            pady=10
        )


        self.real_label_display = ctk.CTkLabel(
            label_frame,
            text="SIN ETIQUETA",
            font=ctk.CTkFont(
                size=15,
                weight="bold"
            )
        )

        self.real_label_display.grid(
            row=0,
            column=1,
            padx=10
        )


        self.btn_manual_no_ship = ctk.CTkButton(
            label_frame,
            text="NO BARCO",
            width=105,
            command=lambda:
                self.set_manual_label(0)
        )

        self.btn_manual_no_ship.grid(
            row=0,
            column=2,
            padx=5
        )


        self.btn_manual_ship = ctk.CTkButton(
            label_frame,
            text="BARCO",
            width=105,
            command=lambda:
                self.set_manual_label(1)
        )

        self.btn_manual_ship.grid(
            row=0,
            column=3,
            padx=5
        )


        self.btn_clear_label = ctk.CTkButton(
            label_frame,
            text="Limpiar",
            width=80,
            command=self.clear_manual_label
        )

        self.btn_clear_label.grid(
            row=0,
            column=4,
            padx=10
        )


    # ========================================================
    # PANEL DERECHO
    # ========================================================

    def create_right_panel(self):

        self.right_panel = ctk.CTkFrame(
            self,
            width=380,
            corner_radius=0
        )

        self.right_panel.grid(
            row=0,
            column=2,
            sticky="nsew"
        )

        self.right_panel.grid_propagate(
            False
        )


        # ----------------------------------------------------
        # PREDICCIÓN
        # ----------------------------------------------------

        ctk.CTkLabel(
            self.right_panel,
            text="PREDICCIÓN",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        ).pack(
            pady=(
                25,
                5
            )
        )


        self.prediction_label = ctk.CTkLabel(
            self.right_panel,
            text="—",
            font=ctk.CTkFont(
                size=36,
                weight="bold"
            )
        )

        self.prediction_label.pack(
            pady=10
        )


        # ----------------------------------------------------
        # PROB BARCO
        # ----------------------------------------------------

        self.prob_ship_label = ctk.CTkLabel(
            self.right_panel,
            text="P(BARCO): —",
            font=ctk.CTkFont(
                size=20
            )
        )

        self.prob_ship_label.pack(
            pady=4
        )


        self.prob_no_ship_label = ctk.CTkLabel(
            self.right_panel,
            text="P(NO BARCO): —",
            font=ctk.CTkFont(
                size=20
            )
        )

        self.prob_no_ship_label.pack(
            pady=4
        )


        self.correct_label = ctk.CTkLabel(
            self.right_panel,
            text="",
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            )
        )

        self.correct_label.pack(
            pady=8
        )


        # ----------------------------------------------------
        # TIEMPOS
        # ----------------------------------------------------

        self.time_label = ctk.CTkLabel(
            self.right_panel,
            text=(
                "Inferencia: —\n"
                "Total: —"
            ),
            justify="left"
        )

        self.time_label.pack(
            pady=10
        )


        separator = ctk.CTkFrame(
            self.right_panel,
            height=2
        )

        separator.pack(
            fill="x",
            padx=20,
            pady=12
        )


        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        ctk.CTkLabel(
            self.right_panel,
            text="EVALUACIÓN ACTUAL",
            font=ctk.CTkFont(
                size=17,
                weight="bold"
            )
        ).pack(
            pady=5
        )


        self.metrics_text = ctk.CTkTextbox(
            self.right_panel,
            width=340,
            height=240
        )

        self.metrics_text.pack(
            padx=20,
            pady=10
        )

        self.metrics_text.configure(
            state="disabled"
        )


        # ----------------------------------------------------
        # VALIDACIÓN MODELO
        # ----------------------------------------------------

        ctk.CTkLabel(
            self.right_panel,
            text="VALIDACIÓN DEL MODELO",
            font=ctk.CTkFont(
                size=17,
                weight="bold"
            )
        ).pack(
            pady=(
                10,
                5
            )
        )


        self.validation_label = ctk.CTkLabel(
            self.right_panel,
            text="Cargando...",
            justify="left",
            anchor="w"
        )

        self.validation_label.pack(
            fill="x",
            padx=25,
            pady=8
        )


        # ----------------------------------------------------
        # ESTADO
        # ----------------------------------------------------

        self.status_label = ctk.CTkLabel(
            self.right_panel,
            text="Listo",
            wraplength=330
        )

        self.status_label.pack(
            side="bottom",
            pady=20
        )


    # ========================================================
    # INICIALIZAR MODELO
    # ========================================================

    def initialize_model(self):

        try:

            self.status_label.configure(
                text="Cargando EfficientNet-B0..."
            )

            self.update_idletasks()


            load_model()

            self.model_info = (
                get_model_info()
            )

            self.model_ready = True


            architecture = (
                self.model_info.get(
                    "architecture",
                    "EfficientNet-B0"
                )
            )

            gpu = (
                self.model_info.get(
                    "gpu",
                    "CPU"
                )
            )

            device = (
                self.model_info.get(
                    "device",
                    "cpu"
                )
            )


            self.model_status.configure(
                text=(
                    f"{architecture}\n"
                    f"Device: {device}\n"
                    f"{gpu}"
                )
            )


            cv_accuracy = (
                self.model_info.get(
                    "cv_accuracy"
                )
            )

            cv_std = (
                self.model_info.get(
                    "cv_std"
                )
            )

            recall = (
                self.model_info.get(
                    "cv_recall"
                )
            )

            auc = (
                self.model_info.get(
                    "cv_auc"
                )
            )


            text = ""


            if cv_accuracy is not None:

                text += (
                    f"5-Fold Accuracy:\n"
                    f"{cv_accuracy * 100:.3f}%"
                )


                if cv_std is not None:

                    text += (
                        f" ± "
                        f"{cv_std * 100:.3f}%"
                    )


                text += "\n\n"


            if recall is not None:

                text += (
                    f"Recall OOF:\n"
                    f"{recall * 100:.2f}%\n\n"
                )


            if auc is not None:

                text += (
                    f"ROC-AUC OOF:\n"
                    f"{auc:.6f}"
                )


            self.validation_label.configure(
                text=text
            )


            self.status_label.configure(
                text="✓ Modelo cargado correctamente"
            )


        except Exception as error:

            self.model_ready = False

            self.model_status.configure(
                text="ERROR AL CARGAR MODELO"
            )

            self.status_label.configure(
                text=str(
                    error
                )
            )

            messagebox.showerror(
                "Error",
                (
                    "No fue posible cargar el modelo.\n\n"
                    f"{error}"
                )
            )


    # ========================================================
    # CARGAR CARPETA
    # ========================================================

    def select_folder(self):

        folder = filedialog.askdirectory(
            title="Seleccionar carpeta de imágenes"
        )


        if not folder:

            return


        folder = Path(
            folder
        )


        images = [

            path

            for path in folder.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]


        images = sorted(
            images,
            key=lambda path:
                path.name.lower()
        )


        if not images:

            messagebox.showwarning(
                "Sin imágenes",
                "No se encontraron imágenes válidas."
            )

            return


        # IMPORTANTE:
        # evitar etiquetas de una carpeta anterior.
        self.csv_labels = {}

        self.manual_labels = {}

        self.results = {}

        self.csv_path = None


        self.current_folder = folder

        self.image_paths = images

        self.current_index = 0


        # ----------------------------------------------------
        # AUTO-DETECTAR CSV
        # ----------------------------------------------------

        candidates = [

            folder
            /
            "labels.csv",

            folder.parent
            /
            "labels.csv",
        ]


        csv_loaded = False


        for candidate in candidates:

            if candidate.exists():

                try:

                    self.load_labels_csv(
                        candidate
                    )

                    csv_loaded = True

                    break

                except Exception:

                    pass


        if not csv_loaded:

            self.labels_status.configure(
                text="Etiquetas: filename/manual"
            )


        self.progress.set(
            0
        )

        self.progress_label.configure(
            text=(
                f"0 / "
                f"{len(self.image_paths)}"
            )
        )


        self.display_current_image()

        self.update_metrics()


        self.status_label.configure(
            text=(
                f"{len(images)} imágenes cargadas"
            )
        )


    # ========================================================
    # SELECCIONAR CSV
    # ========================================================

    def select_csv(self):

        path = filedialog.askopenfilename(

            title="Seleccionar archivo de etiquetas",

            filetypes=[
                (
                    "CSV",
                    "*.csv"
                )
            ]
        )


        if not path:

            return


        try:

            self.load_labels_csv(
                Path(
                    path
                )
            )

            self.refresh_result_labels()

            self.display_current_image()

            self.update_metrics()


        except Exception as error:

            messagebox.showerror(
                "Error CSV",
                str(
                    error
                )
            )


    # ========================================================
    # CARGAR LABELS.CSV
    # ========================================================

    def load_labels_csv(
        self,
        path
    ):

        df = pd.read_csv(
            path
        )


        # ----------------------------------------------------
        # NORMALIZAR COLUMNAS
        # ----------------------------------------------------

        normalized = {

            str(column).strip().lower():
                column

            for column
            in df.columns
        }


        filename_candidates = [
            "filename",
            "file",
            "image",
            "name",
            "imagen",
            "archivo",
        ]


        label_candidates = [
            "label",
            "class",
            "target",
            "etiqueta",
            "clase",
        ]


        filename_column = None

        label_column = None


        for candidate in filename_candidates:

            if candidate in normalized:

                filename_column = normalized[
                    candidate
                ]

                break


        for candidate in label_candidates:

            if candidate in normalized:

                label_column = normalized[
                    candidate
                ]

                break


        if filename_column is None:

            raise ValueError(
                "El CSV no contiene una columna filename."
            )


        if label_column is None:

            raise ValueError(
                "El CSV no contiene una columna label."
            )


        labels = {}


        for _, row in df.iterrows():

            filename = str(
                row[
                    filename_column
                ]
            ).strip()


            label = self.parse_label_value(
                row[
                    label_column
                ]
            )


            if label is not None:

                labels[
                    Path(
                        filename
                    ).name
                ] = label


        self.csv_labels = labels

        self.csv_path = path


        self.labels_status.configure(
            text=(
                f"CSV: {path.name}\n"
                f"{len(labels)} etiquetas"
            )
        )


    # ========================================================
    # PARSEAR LABEL
    # ========================================================

    def parse_label_value(
        self,
        value
    ):

        if pd.isna(
            value
        ):

            return None


        text = str(
            value
        ).strip().lower()


        positives = {
            "1",
            "1.0",
            "ship",
            "barco",
            "yes",
            "true",
        }


        negatives = {
            "0",
            "0.0",
            "no_ship",
            "no ship",
            "noship",
            "no barco",
            "no_barco",
            "false",
            "no",
        }


        if text in positives:

            return 1


        if text in negatives:

            return 0


        try:

            numeric = int(
                float(
                    text
                )
            )


            if numeric in [
                0,
                1
            ]:

                return numeric


        except Exception:

            pass


        return None


    # ========================================================
    # OBTENER ETIQUETA REAL
    # ========================================================

    def get_true_label(
        self,
        path
    ):

        filename = path.name


        # ----------------------------------------------------
        # PRIORIDAD 1: MANUAL
        # ----------------------------------------------------

        if filename in self.manual_labels:

            return (
                self.manual_labels[
                    filename
                ],
                "manual"
            )


        # ----------------------------------------------------
        # PRIORIDAD 2: CSV
        # ----------------------------------------------------

        if filename in self.csv_labels:

            return (
                self.csv_labels[
                    filename
                ],
                "CSV"
            )


        # ----------------------------------------------------
        # PRIORIDAD 3: SHIPSNET FILENAME
        # ----------------------------------------------------

        stem = path.stem


        if "__" in stem:

            first = (
                stem
                .split(
                    "__",
                    1
                )[0]
            )


            if first in [
                "0",
                "1"
            ]:

                return (
                    int(
                        first
                    ),
                    "filename"
                )


        return (
            None,
            None
        )


    # ========================================================
    # LABEL MANUAL
    # ========================================================

    def set_manual_label(
        self,
        label
    ):

        if not self.image_paths:

            return


        path = self.image_paths[
            self.current_index
        ]


        self.manual_labels[
            path.name
        ] = label


        self.refresh_result_labels()

        self.display_current_image()

        self.update_metrics()


    def clear_manual_label(self):

        if not self.image_paths:

            return


        path = self.image_paths[
            self.current_index
        ]


        self.manual_labels.pop(
            path.name,
            None
        )


        self.refresh_result_labels()

        self.display_current_image()

        self.update_metrics()


    # ========================================================
    # ACTUALIZAR LABELS EN RESULTADOS
    # ========================================================

    def refresh_result_labels(self):

        for key, result in self.results.items():

            path = Path(
                result[
                    "path"
                ]
            )


            true_label, source = (
                self.get_true_label(
                    path
                )
            )


            result[
                "real_label"
            ] = true_label

            result[
                "label_source"
            ] = source


            if true_label is not None:

                result[
                    "correct"
                ] = (
                    result[
                        "prediction"
                    ]
                    ==
                    true_label
                )

            else:

                result[
                    "correct"
                ] = None


    # ========================================================
    # MOSTRAR IMAGEN
    # ========================================================

    def display_current_image(self):

        if not self.image_paths:

            return


        path = self.image_paths[
            self.current_index
        ]


        self.filename_label.configure(
            text=path.name
        )


        self.position_label.configure(
            text=(
                f"{self.current_index + 1}"
                f" / "
                f"{len(self.image_paths)}"
            )
        )


        # ----------------------------------------------------
        # CARGAR PREVIEW
        # ----------------------------------------------------

        try:

            image = Image.open(
                path
            ).convert(
                "RGB"
            )


            preview_size = (
                560,
                560
            )


            image.thumbnail(
                preview_size,
                Image.Resampling.LANCZOS
            )


            self.preview_ctk_image = (
                ctk.CTkImage(

                    light_image=image,

                    dark_image=image,

                    size=(
                        image.width,
                        image.height
                    )
                )
            )


            self.image_label.configure(

                image=self.preview_ctk_image,

                text=""
            )


        except Exception as error:

            self.image_label.configure(
                image=None,
                text=(
                    "ERROR AL ABRIR IMAGEN\n"
                    f"{error}"
                )
            )


        # ----------------------------------------------------
        # ETIQUETA REAL
        # ----------------------------------------------------

        true_label, source = (
            self.get_true_label(
                path
            )
        )


        if true_label == 1:

            text = "BARCO"


        elif true_label == 0:

            text = "NO BARCO"


        else:

            text = "SIN ETIQUETA"


        if source:

            text += (
                f"  ({source})"
            )


        self.real_label_display.configure(
            text=text
        )


        # ----------------------------------------------------
        # RESULTADO EXISTENTE
        # ----------------------------------------------------

        key = str(
            path.resolve()
        )


        if key in self.results:

            self.display_prediction_result(
                self.results[
                    key
                ]
            )

        else:

            self.clear_prediction_display()


    # ========================================================
    # NAVEGACIÓN
    # ========================================================

    def previous_image(self):

        if not self.image_paths:

            return


        self.current_index = max(
            0,
            self.current_index - 1
        )


        self.display_current_image()


    def next_image(self):

        if not self.image_paths:

            return


        self.current_index = min(

            len(
                self.image_paths
            ) - 1,

            self.current_index + 1
        )


        self.display_current_image()


    # ========================================================
    # LIMPIAR RESULTADO
    # ========================================================

    def clear_prediction_display(self):

        self.prediction_label.configure(
            text="—"
        )

        self.prob_ship_label.configure(
            text="P(BARCO): —"
        )

        self.prob_no_ship_label.configure(
            text="P(NO BARCO): —"
        )

        self.correct_label.configure(
            text=""
        )

        self.time_label.configure(
            text=(
                "Inferencia: —\n"
                "Total: —"
            )
        )


    # ========================================================
    # PREDICCIÓN INDIVIDUAL
    # ========================================================

    def predict_current(self):

        if not self.model_ready:

            messagebox.showwarning(
                "Modelo",
                "El modelo todavía no está listo."
            )

            return


        if not self.image_paths:

            messagebox.showwarning(
                "Imágenes",
                "Primero carga una carpeta."
            )

            return


        if self.processing:

            return


        path = self.image_paths[
            self.current_index
        ]


        self.status_label.configure(
            text=(
                f"Clasificando {path.name}..."
            )
        )


        self.update_idletasks()


        try:

            output = predict_image(

                path,

                use_tta=bool(
                    self.tta_var.get()
                )
            )


            true_label, source = (
                self.get_true_label(
                    path
                )
            )


            result = self.create_result_record(

                path,

                output,

                true_label,

                source
            )


            key = str(
                path.resolve()
            )


            self.results[
                key
            ] = result


            self.display_prediction_result(
                result
            )


            self.update_metrics()


            self.status_label.configure(
                text="✓ Clasificación completada"
            )


        except Exception as error:

            messagebox.showerror(
                "Error de inferencia",
                str(
                    error
                )
            )


            self.status_label.configure(
                text="Error de inferencia"
            )


    # ========================================================
    # CREAR RESULT RECORD
    # ========================================================

    def create_result_record(
        self,
        path,
        output,
        true_label,
        source
    ):

        prediction = int(
            output[
                "prediction"
            ]
        )


        correct = None


        if true_label is not None:

            correct = (
                prediction
                ==
                true_label
            )


        return {

            "filename":
                path.name,

            "path":
                str(
                    path.resolve()
                ),

            "real_label":
                true_label,

            "label_source":
                source,

            "prediction":
                prediction,

            "predicted_class":
                output[
                    "class_name"
                ],

            "prob_ship":
                float(
                    output[
                        "prob_ship"
                    ]
                ),

            "prob_no_ship":
                float(
                    output[
                        "prob_no_ship"
                    ]
                ),

            "threshold":
                float(
                    output[
                        "threshold"
                    ]
                ),

            "correct":
                correct,

            "use_tta":
                bool(
                    output[
                        "use_tta"
                    ]
                ),

            "tta_std":
                float(
                    output.get(
                        "tta_std",
                        0.0
                    )
                ),

            "preprocessing_ms":
                float(
                    output.get(
                        "preprocessing_ms",
                        0.0
                    )
                ),

            "inference_ms":
                float(
                    output.get(
                        "inference_ms",
                        0.0
                    )
                ),

            "total_ms":
                float(
                    output.get(
                        "total_ms",
                        0.0
                    )
                ),
        }


    # ========================================================
    # MOSTRAR RESULTADO
    # ========================================================

    def display_prediction_result(
        self,
        result
    ):

        self.prediction_label.configure(
            text=result[
                "predicted_class"
            ]
        )


        self.prob_ship_label.configure(
            text=(
                f"P(BARCO): "
                f"{result['prob_ship'] * 100:.2f}%"
            )
        )


        self.prob_no_ship_label.configure(
            text=(
                f"P(NO BARCO): "
                f"{result['prob_no_ship'] * 100:.2f}%"
            )
        )


        if result[
            "correct"
        ] is True:

            self.correct_label.configure(
                text="✓ CORRECTO"
            )


        elif result[
            "correct"
        ] is False:

            self.correct_label.configure(
                text="✗ ERROR"
            )


        else:

            self.correct_label.configure(
                text="Sin etiqueta real"
            )


        mode = (
            "TTA"
            if result[
                "use_tta"
            ]
            else
            "Normal"
        )


        self.time_label.configure(

            text=(

                f"Modo: {mode}\n"

                f"Inferencia: "
                f"{result['inference_ms']:.2f} ms\n"

                f"Total: "
                f"{result['total_ms']:.2f} ms"
            )
        )


    # ========================================================
    # INICIAR PROCESAMIENTO CARPETA
    # ========================================================

    def start_folder_processing(self):

        if not self.model_ready:

            messagebox.showwarning(
                "Modelo",
                "El modelo no está listo."
            )

            return


        if not self.image_paths:

            messagebox.showwarning(
                "Imágenes",
                "Carga primero una carpeta."
            )

            return


        if self.processing:

            return


        self.processing = True

        self.stop_event.clear()


        self.btn_process_all.configure(
            state="disabled"
        )

        self.btn_predict.configure(
            state="disabled"
        )

        self.btn_folder.configure(
            state="disabled"
        )

        self.btn_stop.configure(
            state="normal"
        )


        use_tta = bool(
            self.tta_var.get()
        )


        paths_snapshot = list(
            self.image_paths
        )


        self.progress.set(
            0
        )


        self.status_label.configure(
            text=(
                "Procesando carpeta..."
            )
        )


        self.processing_thread = (
            threading.Thread(

                target=self.folder_worker,

                args=(
                    paths_snapshot,
                    use_tta
                ),

                daemon=True
            )
        )


        self.processing_thread.start()


    # ========================================================
    # WORKER
    # ========================================================

    def folder_worker(
        self,
        paths,
        use_tta
    ):

        start = (
            time.perf_counter()
        )


        processed = 0


        for index, path in enumerate(
            paths
        ):

            if self.stop_event.is_set():

                break


            try:

                output = predict_image(

                    path,

                    use_tta=use_tta
                )


                true_label, source = (
                    self.get_true_label(
                        path
                    )
                )


                result = self.create_result_record(

                    path,

                    output,

                    true_label,

                    source
                )


                self.worker_queue.put(

                    (
                        "result",
                        index,
                        len(
                            paths
                        ),
                        result
                    )
                )


                processed += 1


            except Exception as error:

                self.worker_queue.put(

                    (
                        "error",
                        path.name,
                        str(
                            error
                        )
                    )
                )


        elapsed = (
            time.perf_counter()
            -
            start
        )


        self.worker_queue.put(

            (
                "done",
                processed,
                len(
                    paths
                ),
                elapsed,
                self.stop_event.is_set()
            )
        )


    # ========================================================
    # POLL QUEUE
    # ========================================================

    def poll_worker_queue(self):

        try:

            while True:

                message = (
                    self.worker_queue
                    .get_nowait()
                )


                kind = message[
                    0
                ]


                # ============================================
                # RESULT
                # ============================================

                if kind == "result":

                    _, index, total, result = (
                        message
                    )


                    key = str(
                        Path(
                            result[
                                "path"
                            ]
                        ).resolve()
                    )


                    self.results[
                        key
                    ] = result


                    completed = (
                        index + 1
                    )


                    fraction = (
                        completed
                        /
                        total
                    )


                    self.progress.set(
                        fraction
                    )


                    self.progress_label.configure(

                        text=(
                            f"{completed}"
                            f" / "
                            f"{total}"
                        )
                    )


                    self.status_label.configure(

                        text=(
                            f"Procesando "
                            f"{completed}/{total}"
                        )
                    )


                    # Actualizar la imagen mostrada
                    # si corresponde al resultado.
                    if self.image_paths:

                        current_path = (
                            self.image_paths[
                                self.current_index
                            ]
                        )


                        if (
                            current_path.name
                            ==
                            result[
                                "filename"
                            ]
                        ):

                            self.display_prediction_result(
                                result
                            )


                    self.update_metrics()


                # ============================================
                # ERROR
                # ============================================

                elif kind == "error":

                    _, filename, error = (
                        message
                    )


                    print(
                        f"Error en "
                        f"{filename}: "
                        f"{error}"
                    )


                # ============================================
                # DONE
                # ============================================

                elif kind == "done":

                    (
                        _,
                        processed,
                        total,
                        elapsed,
                        stopped
                    ) = message


                    self.finish_processing(

                        processed,
                        total,
                        elapsed,
                        stopped
                    )


        except queue.Empty:

            pass


        self.after(
            100,
            self.poll_worker_queue
        )


    # ========================================================
    # FINALIZAR PROCESAMIENTO
    # ========================================================

    def finish_processing(
        self,
        processed,
        total,
        elapsed,
        stopped
    ):

        self.processing = False


        self.btn_process_all.configure(
            state="normal"
        )

        self.btn_predict.configure(
            state="normal"
        )

        self.btn_folder.configure(
            state="normal"
        )

        self.btn_stop.configure(
            state="disabled"
        )


        if stopped:

            self.status_label.configure(
                text=(
                    f"Procesamiento detenido "
                    f"({processed}/{total})"
                )
            )


        else:

            self.progress.set(
                1
            )


            self.progress_label.configure(
                text=(
                    f"{processed}"
                    f" / "
                    f"{total}"
                )
            )


            self.status_label.configure(
                text=(
                    f"✓ {processed} imágenes "
                    f"procesadas en "
                    f"{elapsed:.2f} s"
                )
            )


        self.update_metrics()


    # ========================================================
    # DETENER
    # ========================================================

    def stop_processing(self):

        if self.processing:

            self.stop_event.set()

            self.status_label.configure(
                text="Deteniendo..."
            )


    # ========================================================
    # OBTENER DATAFRAME DE RESULTADOS
    # ========================================================

    def get_results_dataframe(self):

        if not self.results:

            return pd.DataFrame()


        rows = list(
            self.results.values()
        )


        df = pd.DataFrame(
            rows
        )


        return df


    # ========================================================
    # MÉTRICAS
    # ========================================================

    def calculate_current_metrics(self):

        df = (
            self.get_results_dataframe()
        )


        if df.empty:

            return None


        labeled = df[

            df[
                "real_label"
            ].notna()

        ].copy()


        if labeled.empty:

            return {

                "processed":
                    len(
                        df
                    ),

                "labeled":
                    0,

                "unlabeled":
                    len(
                        df
                    ),

                "mean_inference_ms":
                    df[
                        "inference_ms"
                    ].mean(),

                "mean_total_ms":
                    df[
                        "total_ms"
                    ].mean(),
            }


        y_true = (
            labeled[
                "real_label"
            ]
            .astype(
                int
            )
            .to_numpy()
        )


        y_pred = (
            labeled[
                "prediction"
            ]
            .astype(
                int
            )
            .to_numpy()
        )


        probabilities = (
            labeled[
                "prob_ship"
            ]
            .astype(
                float
            )
            .to_numpy()
        )


        accuracy = accuracy_score(
            y_true,
            y_pred
        )


        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )


        balanced = (
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        )


        try:

            auc = roc_auc_score(
                y_true,
                probabilities
            )

        except Exception:

            auc = None


        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1
            ]
        )


        return {

            "processed":
                len(
                    df
                ),

            "labeled":
                len(
                    labeled
                ),

            "unlabeled":
                len(
                    df
                )
                -
                len(
                    labeled
                ),

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "balanced_accuracy":
                balanced,

            "roc_auc":
                auc,

            "confusion_matrix":
                cm,

            "mean_inference_ms":
                df[
                    "inference_ms"
                ].mean(),

            "mean_total_ms":
                df[
                    "total_ms"
                ].mean(),
        }


    # ========================================================
    # ACTUALIZAR MÉTRICAS
    # ========================================================

    def update_metrics(self):

        metrics = (
            self.calculate_current_metrics()
        )


        if metrics is None:

            text = (
                "Procesadas: 0\n"
                "Con etiqueta: 0\n\n"
                "Accuracy: —\n"
                "Precision: —\n"
                "Recall: —\n"
                "F1: —"
            )


        elif (
            metrics[
                "labeled"
            ]
            ==
            0
        ):

            text = (

                f"Procesadas: "
                f"{metrics['processed']}\n"

                f"Con etiqueta: 0\n"

                f"Sin etiqueta: "
                f"{metrics['unlabeled']}\n\n"

                f"Tiempo inferencia medio:\n"
                f"{metrics['mean_inference_ms']:.2f} ms\n\n"

                "No hay etiquetas reales.\n"
                "Se realiza solo inferencia."
            )


        else:

            auc_text = "—"


            if (
                metrics[
                    "roc_auc"
                ]
                is not None
            ):

                auc_text = (
                    f"{metrics['roc_auc']:.4f}"
                )


            text = (

                f"Procesadas: "
                f"{metrics['processed']}\n"

                f"Evaluadas: "
                f"{metrics['labeled']}\n"

                f"Sin etiqueta: "
                f"{metrics['unlabeled']}\n\n"

                f"Accuracy: "
                f"{metrics['accuracy'] * 100:.2f}%\n"

                f"Precision: "
                f"{metrics['precision'] * 100:.2f}%\n"

                f"Recall: "
                f"{metrics['recall'] * 100:.2f}%\n"

                f"F1: "
                f"{metrics['f1'] * 100:.2f}%\n"

                f"Balanced Acc: "
                f"{metrics['balanced_accuracy'] * 100:.2f}%\n"

                f"ROC-AUC: "
                f"{auc_text}\n\n"

                f"Inferencia media: "
                f"{metrics['mean_inference_ms']:.2f} ms\n"

                f"Tiempo total medio: "
                f"{metrics['mean_total_ms']:.2f} ms"
            )


        self.metrics_text.configure(
            state="normal"
        )


        self.metrics_text.delete(
            "1.0",
            "end"
        )


        self.metrics_text.insert(
            "1.0",
            text
        )


        self.metrics_text.configure(
            state="disabled"
        )


    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    def show_confusion_matrix(self):

        metrics = (
            self.calculate_current_metrics()
        )


        if (
            metrics is None
            or
            metrics.get(
                "labeled",
                0
            )
            ==
            0
        ):

            messagebox.showinfo(
                "Matriz",
                (
                    "No hay suficientes imágenes "
                    "con etiqueta real."
                )
            )

            return


        cm = metrics[
            "confusion_matrix"
        ]


        window = ctk.CTkToplevel(
            self
        )

        window.title(
            "Matriz de Confusión"
        )

        window.geometry(
            "620x620"
        )


        figure = Figure(
            figsize=(
                5.5,
                5.2
            ),
            dpi=100
        )


        ax = figure.add_subplot(
            111
        )


        image = ax.imshow(
            cm
        )


        ax.set_xticks(
            [
                0,
                1
            ]
        )

        ax.set_yticks(
            [
                0,
                1
            ]
        )


        ax.set_xticklabels(
            [
                "NO BARCO",
                "BARCO"
            ]
        )

        ax.set_yticklabels(
            [
                "NO BARCO",
                "BARCO"
            ]
        )


        ax.set_xlabel(
            "Predicción"
        )

        ax.set_ylabel(
            "Etiqueta real"
        )

        ax.set_title(
            "Matriz de Confusión"
        )


        for i in range(
            2
        ):

            for j in range(
                2
            ):

                ax.text(
                    j,
                    i,
                    str(
                        cm[
                            i,
                            j
                        ]
                    ),
                    ha="center",
                    va="center",
                    fontsize=18
                )


        figure.tight_layout()


        canvas = FigureCanvasTkAgg(
            figure,
            master=window
        )


        canvas.draw()


        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )


    # ========================================================
    # EXPORTAR
    # ========================================================

    def export_results(self):

        df = (
            self.get_results_dataframe()
        )


        if df.empty:

            messagebox.showinfo(
                "Exportar",
                "No hay resultados para exportar."
            )

            return


        output_path = filedialog.asksaveasfilename(

            title="Guardar resultados",

            defaultextension=".csv",

            filetypes=[
                (
                    "CSV",
                    "*.csv"
                )
            ],

            initialfile=(
                "resultados_clasificador_barcos.csv"
            )
        )


        if not output_path:

            return


        # ----------------------------------------------------
        # ORDEN DE COLUMNAS
        # ----------------------------------------------------

        preferred_columns = [

            "filename",

            "real_label",

            "label_source",

            "prediction",

            "predicted_class",

            "prob_ship",

            "prob_no_ship",

            "threshold",

            "correct",

            "use_tta",

            "tta_std",

            "preprocessing_ms",

            "inference_ms",

            "total_ms",

            "path",
        ]


        columns = [

            column

            for column
            in preferred_columns

            if column
            in df.columns
        ]


        df[
            columns
        ].to_csv(

            output_path,

            index=False
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        metrics = (
            self.calculate_current_metrics()
        )


        summary_path = (

            Path(
                output_path
            ).with_name(

                Path(
                    output_path
                ).stem

                +

                "_summary.csv"
            )
        )


        if metrics:

            summary_data = {}


            for key, value in metrics.items():

                if key == "confusion_matrix":

                    cm = value

                    summary_data[
                        "TN"
                    ] = int(
                        cm[
                            0,
                            0
                        ]
                    )

                    summary_data[
                        "FP"
                    ] = int(
                        cm[
                            0,
                            1
                        ]
                    )

                    summary_data[
                        "FN"
                    ] = int(
                        cm[
                            1,
                            0
                        ]
                    )

                    summary_data[
                        "TP"
                    ] = int(
                        cm[
                            1,
                            1
                        ]
                    )

                else:

                    summary_data[
                        key
                    ] = value


            pd.DataFrame(
                [
                    summary_data
                ]
            ).to_csv(

                summary_path,

                index=False
            )


        messagebox.showinfo(

            "Exportación completada",

            (
                "Resultados guardados en:\n"
                f"{output_path}\n\n"

                "Resumen guardado en:\n"
                f"{summary_path}"
            )
        )


    # ========================================================
    # CIERRE
    # ========================================================

    def on_close(self):

        if self.processing:

            self.stop_event.set()


        self.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    app = ShipClassifierApp()

    app.protocol(
        "WM_DELETE_WINDOW",
        app.on_close
    )

    app.mainloop()


if __name__ == "__main__":

    main()