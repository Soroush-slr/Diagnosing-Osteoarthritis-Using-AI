import os
import sys
import traceback
import tempfile
import subprocess
import webbrowser
from datetime import datetime
from tkinter import filedialog, messagebox

import customtkinter as ctk
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from PIL import Image

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
    )

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

APP_TITLE = "AI Joint Disease Screening System"
APP_SIZE = "1540x950"

MODEL_FILE = "best_questionnaire_model.joblib"
FEATURE_FILE = "feature_columns.joblib"
LABEL_FILE = "label_encoder.joblib"
LOG_FILE = "application_error.log"
DEFAULT_LOGO_FILE = "clinic_logo.png"

FEATURES = [
    "Joint Symptoms",
    "Joint pain",
    "Impact of Activity",
    "Morning Stiffness",
    "Joint Functionality",
    "Functional Limitations",
    "Swelling and Inflammation",
    "Perceptible Changes in Joints",
    "History and Progression of Symptoms",
    "Symptom Fluctuation",
    "Family History",
    "Body part affected",
]

QUESTIONS = {
    "Joint Symptoms": "Are your symptoms generally symmetrical on both sides?",
    "Joint pain": "Do you have pain or stiffness in non-weight-bearing joints?",
    "Impact of Activity": "Does pain increase after prolonged activity?",
    "Morning Stiffness": "Does morning stiffness improve within 10-15 minutes?",
    "Joint Functionality": "Do daily tasks become difficult when symptoms worsen?",
    "Functional Limitations": "Do you need to modify hand grip because of discomfort?",
    "Swelling and Inflammation": "Have you noticed enlargement or hard lumps in joints?",
    "Perceptible Changes in Joints": "Have your fingers or hands gradually deformed?",
    "History and Progression of Symptoms": "Has any joint gradually lost range of motion over time?",
    "Symptom Fluctuation": "Do symptoms improve for a while and then return?",
    "Family History": "Is there a family history of joint disease?",
    "Body part affected": "Are symptoms affecting a repeated pattern of joints?",
}

ANSWER_MAP = {"No": 0, "I don't know": 1, "Yes": 2}
REVERSE_ANSWER_MAP = {0: "No", 1: "I don't know", 2: "Yes"}


def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)


def get_documents_directory():
    home = os.path.expanduser("~")
    documents = os.path.join(home, "Documents")
    return documents if os.path.isdir(documents) else home


def get_app_records_directory():
    path = os.path.join(get_documents_directory(), "AI_Joint_Screening_Records")
    os.makedirs(path, exist_ok=True)
    return path


def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_report_filename(ext):
    return os.path.join(
        get_app_records_directory(),
        f"joint_screening_report_{make_timestamp()}.{ext}",
    )


def make_excel_filename():
    return os.path.join(get_app_records_directory(), "patient_screening_records.xlsx")


def write_error_log(context, exc):
    try:
        path = os.path.join(get_app_records_directory(), LOG_FILE)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Error: {exc}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass


def open_with_default_app(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


class SummaryCard(ctk.CTkFrame):
    def __init__(self, master, title, color):
        super().__init__(
            master,
            corner_radius=16,
            fg_color="white",
            border_width=1,
            border_color="#D8E4EE",
        )
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#64748B",
        )
        self.title_label.pack(anchor="w", padx=14, pady=(12, 4))
        self.value_label = ctk.CTkLabel(
            self,
            text="--",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=color,
        )
        self.value_label.pack(anchor="w", padx=14, pady=(0, 12))

    def set_value(self, value, color=None):
        self.value_label.configure(text=value)
        if color:
            self.value_label.configure(text_color=color)


class DiagnosisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_SIZE)
        self.minsize(1380, 860)
        self.configure(fg_color="#EAF1F8")

        self.model = None
        self.feature_columns = FEATURES.copy()
        self.label_encoder = None

        self.answer_boxes = {}
        self.logo_path = ""
        self.logo_ctk_image = None

        self.last_result_text = ""
        self.last_html_report = ""
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_answers = None
        self.last_probabilities = []
        self.last_probability_labels = []
        self.last_symptom_score = 0
        self.last_risk_level = "Not calculated"

        self.chart_canvas = None
        self.chart_figure = None

        self.load_model_files()
        self.build_ui()
        self.try_load_default_logo()

    def load_model_files(self):
        try:
            model_path = resource_path(MODEL_FILE)
            feature_path = resource_path(FEATURE_FILE)
            label_path = resource_path(LABEL_FILE)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = joblib.load(model_path)

            if os.path.exists(feature_path):
                self.feature_columns = joblib.load(feature_path)

            if os.path.exists(label_path):
                self.label_encoder = joblib.load(label_path)

        except Exception as exc:
            write_error_log("load_model_files", exc)
            messagebox.showerror("Startup Error", str(exc))
            raise

    def build_ui(self):
        self.build_header()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.left_panel = ctk.CTkScrollableFrame(
            body,
            corner_radius=18,
            fg_color="#F8FBFE",
            border_width=1,
            border_color="#D8E4EE",
        )
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.right_panel = ctk.CTkScrollableFrame(
            body,
            width=620,
            corner_radius=18,
            fg_color="#F8FBFE",
            border_width=1,
            border_color="#D8E4EE",
        )
        self.right_panel.pack(side="right", fill="both", expand=False, padx=(8, 0))

        self.build_patient_section()
        self.build_questionnaire_section()

        self.build_controls()
        self.build_branding_section()
        self.build_dashboard_cards()
        self.build_confidence_panel()
        self.build_chart_panel()
        self.build_result_area()
        self.build_footer_note()

    def build_header(self):
        header = ctk.CTkFrame(self, corner_radius=22, fg_color="#0B3A63")
        header.pack(fill="x", padx=16, pady=16)

        top = ctk.CTkFrame(header, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(18, 8))

        ctk.CTkLabel(
            top,
            text="AI Joint Disease Screening System",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="white",
        ).pack(side="left", anchor="w")

        ctk.CTkLabel(
            top,
            text="CLINICAL AI SUPPORT",
            fg_color="#E8F3FF",
            text_color="#0B3A63",
            corner_radius=999,
            font=ctk.CTkFont(size=13, weight="bold"),
            padx=14,
            pady=8,
        ).pack(side="right", anchor="e")

        ctk.CTkLabel(
            header,
            text="Professional questionnaire-based screening support for joint disease evaluation",
            font=ctk.CTkFont(size=14),
            text_color="#D7E5F3",
        ).pack(anchor="w", padx=20, pady=(0, 18))

    def build_patient_section(self):
        frame = ctk.CTkFrame(self.left_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(
            frame,
            text="Patient Information",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 10))

        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", padx=16, pady=(0, 10))
        row1.grid_columnconfigure((0, 1, 2), weight=1)

        self.patient_name = ctk.CTkEntry(row1, placeholder_text="Patient Name", height=42)
        self.patient_name.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.patient_age = ctk.CTkEntry(row1, placeholder_text="Age", height=42)
        self.patient_age.grid(row=0, column=1, padx=8, sticky="ew")

        self.patient_gender = ctk.CTkComboBox(
            row1,
            values=["Female", "Male", "Other"],
            state="readonly",
            height=42,
        )
        self.patient_gender.set("Female")
        self.patient_gender.grid(row=0, column=2, padx=(8, 0), sticky="ew")

        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", padx=16, pady=(0, 14))
        row2.grid_columnconfigure((0, 1), weight=1)

        self.patient_id = ctk.CTkEntry(row2, placeholder_text="Patient ID / File Number", height=42)
        self.patient_id.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.clinic_name = ctk.CTkEntry(row2, placeholder_text="Clinic / Department", height=42)
        self.clinic_name.grid(row=0, column=1, padx=(8, 0), sticky="ew")

    def build_questionnaire_section(self):
        ctk.CTkLabel(
            self.left_panel,
            text="Clinical Questionnaire",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=14, pady=(4, 8))

        ctk.CTkLabel(
            self.left_panel,
            text="Answer each item carefully. This tool supports screening and does not replace physician judgment.",
            font=ctk.CTkFont(size=13),
            text_color="#607284",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        for idx, feature in enumerate(FEATURES, start=1):
            box = ctk.CTkFrame(self.left_panel, corner_radius=16, fg_color="white")
            box.pack(fill="x", padx=12, pady=6)

            ctk.CTkLabel(
                box,
                text=f"{idx}. {feature}",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#0F4C81",
            ).pack(anchor="w", padx=16, pady=(12, 4))

            ctk.CTkLabel(
                box,
                text=QUESTIONS[feature],
                wraplength=780,
                justify="left",
                text_color="#1F2937",
            ).pack(anchor="w", padx=16, pady=(0, 8))

            combo = ctk.CTkComboBox(
                box,
                values=["No", "I don't know", "Yes"],
                state="readonly",
                width=240,
                height=38,
            )
            combo.set("I don't know")
            combo.pack(anchor="w", padx=16, pady=(0, 12))
            self.answer_boxes[feature] = combo

    def build_controls(self):
        frame = ctk.CTkFrame(self.right_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(
            frame,
            text="Control Panel",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 10))

        ctk.CTkLabel(
            frame,
            text="All control buttons are placed here intentionally so they remain visible without scrolling deep into the report.",
            font=ctk.CTkFont(size=12),
            text_color="#64748B",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 14))
        grid.grid_columnconfigure((0, 1), weight=1)

        buttons = [
            ("Analyze Patient", "#0F4C81", self.analyze_patient),
            ("Clear Results Only", "#F59E0B", self.clear_results_only),
            ("Reset Full Form", "#DC2626", self.reset_full_form),
            ("Preview Current Answers", "#7C3AED", self.preview_current_answers),
            ("Show Last Result Again", "#0284C7", self.show_last_result_again),
            ("Save Patient Record to Excel", "#1F7A53", self.save_patient_record_to_excel),
            ("Export TXT Report", "#334155", self.export_txt_report),
            ("Export HTML Report", "#475569", self.export_html_report),
            ("Export PDF Report", "#7C2D12", self.export_pdf_report),
            ("Open Records Folder", "#0F766E", self.open_records_folder),
            ("Print Last Report", "#92400E", self.print_last_report),
            ("Select Clinic Logo", "#5B21B6", self.select_clinic_logo),
        ]

        for idx, (text, color, command) in enumerate(buttons):
            row = idx // 2
            col = idx % 2
            btn = ctk.CTkButton(
                grid,
                text=text,
                command=command,
                height=48,
                corner_radius=12,
                fg_color=color,
                hover_color=color,
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="ew")

    def build_branding_section(self):
        frame = ctk.CTkFrame(self.right_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="Clinic Branding",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 8))

        self.logo_preview_label = ctk.CTkLabel(
            frame,
            text="No clinic logo selected",
            text_color="#64748B",
        )
        self.logo_preview_label.pack(anchor="w", padx=16, pady=(0, 10))

        self.logo_image_label = ctk.CTkLabel(frame, text="")
        self.logo_image_label.pack(anchor="w", padx=16, pady=(0, 14))

    def build_dashboard_cards(self):
        frame = ctk.CTkFrame(self.right_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="AI Clinical Analysis Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 12))

        cards = ctk.CTkFrame(frame, fg_color="transparent")
        cards.pack(fill="x", padx=12, pady=(0, 14))
        cards.grid_columnconfigure((0, 1), weight=1)

        self.prediction_card = SummaryCard(cards, "Prediction", "#0F4C81")
        self.prediction_card.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        self.confidence_card = SummaryCard(cards, "Confidence", "#0E7490")
        self.confidence_card.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        self.score_card = SummaryCard(cards, "Symptom Score", "#9333EA")
        self.score_card.grid(row=1, column=0, padx=6, pady=6, sticky="ew")

        self.risk_card = SummaryCard(cards, "Risk Level", "#DC2626")
        self.risk_card.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

    def build_confidence_panel(self):
        frame = ctk.CTkFrame(self.right_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="Confidence Panel",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 10))

        self.confidence_value_label = ctk.CTkLabel(
            frame,
            text="0%",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#0E7490",
        )
        self.confidence_value_label.pack(anchor="w", padx=16)

        self.confidence_progress = ctk.CTkProgressBar(frame, height=18, progress_color="#0E7490")
        self.confidence_progress.pack(fill="x", padx=16, pady=(8, 14))
        self.confidence_progress.set(0)

    def build_chart_panel(self):
        frame = ctk.CTkFrame(self.right_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="Probability Distribution",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 8))

        self.chart_host = ctk.CTkFrame(frame, fg_color="white", height=260)
        self.chart_host.pack(fill="x", padx=12, pady=(0, 12))
        self.chart_host.pack_propagate(False)

        self.render_probability_chart(["No Data"], [0.0])

    def build_result_area(self):
        frame = ctk.CTkFrame(self.right_panel, corner_radius=16, fg_color="white")
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="Detailed Clinical Report",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#123B5D",
        ).pack(anchor="w", padx=16, pady=(14, 10))

        self.result_box = ctk.CTkTextbox(
            frame,
            height=340,
            corner_radius=12,
            wrap="word",
            font=ctk.CTkFont(size=14),
        )
        self.result_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.set_result_text("Ready for analysis...\nFill the form and click Analyze Patient.")

    def build_footer_note(self):
        frame = ctk.CTkFrame(
            self.left_panel,
            corner_radius=16,
            fg_color="#FFF7ED",
            border_width=1,
            border_color="#F6D7B0",
        )
        frame.pack(fill="x", padx=12, pady=(4, 14))

        ctk.CTkLabel(
            frame,
            text="Medical Disclaimer",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#9A3412",
        ).pack(anchor="w", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            frame,
            text=(
                "This application provides AI-assisted screening support. "
                "It is not a substitute for physician judgment, examination, imaging, laboratory testing, or formal diagnosis."
            ),
            wraplength=800,
            justify="left",
            text_color="#7C2D12",
        ).pack(anchor="w", padx=16, pady=(0, 14))

    def try_load_default_logo(self):
        try:
            path = resource_path(DEFAULT_LOGO_FILE)
            if os.path.exists(path):
                self.set_logo(path)
        except Exception as exc:
            write_error_log("try_load_default_logo", exc)

    def select_clinic_logo(self):
        path = filedialog.askopenfilename(
            title="Select Clinic Logo",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")],
        )
        if path:
            self.set_logo(path)

    def set_logo(self, path):
        try:
            image = Image.open(path)
            image.thumbnail((220, 90))
            self.logo_ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.logo_image_label.configure(image=self.logo_ctk_image, text="")
            self.logo_preview_label.configure(text=os.path.basename(path))
            self.logo_path = path
        except Exception as exc:
            write_error_log("set_logo", exc)
            messagebox.showerror("Logo Error", str(exc))

    def validate_age(self):
        age_text = self.patient_age.get().strip()
        if not age_text:
            return ""
        try:
            age = int(age_text)
        except ValueError:
            raise ValueError("Age must be a valid integer.")
        if age < 0 or age > 120:
            raise ValueError("Age must be between 0 and 120.")
        return age

    def collect_answers(self):
        return {feature: ANSWER_MAP[self.answer_boxes[feature].get()] for feature in FEATURES}

    def calculate_symptom_score(self, answers):
        return sum(answers.values())

    def calculate_risk_level(self, confidence, score):
        if confidence >= 0.85 or score >= 18:
            return "High"
        if confidence >= 0.65 or score >= 12:
            return "Moderate"
        return "Low"

    def get_risk_color(self, risk):
        if risk == "High":
            return "#DC2626"
        if risk == "Moderate":
            return "#D97706"
        return "#15803D"

    def get_prediction_display(self, pred):
        if self.label_encoder is not None:
            try:
                return str(self.label_encoder.inverse_transform([pred])[0])
            except Exception:
                return str(pred)
        return str(pred)

    def get_class_labels(self):
        if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            return [str(x) for x in self.label_encoder.classes_]
        if hasattr(self.model, "classes_"):
            return [str(x) for x in self.model.classes_]
        return ["Class 1", "Class 2", "Class 3"]

    def update_dashboard(self, prediction, confidence, score, risk):
        risk_color = self.get_risk_color(risk)
        self.prediction_card.set_value(prediction, "#0F4C81")
        self.confidence_card.set_value(f"{confidence:.1%}", "#0E7490")
        self.score_card.set_value(str(score), "#9333EA")
        self.risk_card.set_value(risk, risk_color)
        self.confidence_value_label.configure(text=f"{confidence:.1%}")
        self.confidence_progress.set(confidence)

    def clear_dashboard(self):
        self.prediction_card.set_value("--", "#0F4C81")
        self.confidence_card.set_value("--", "#0E7490")
        self.score_card.set_value("--", "#9333EA")
        self.risk_card.set_value("--", "#DC2626")
        self.confidence_value_label.configure(text="0%")
        self.confidence_progress.set(0)

    def set_result_text(self, text):
        self.result_box.delete("1.0", "end")
        self.result_box.insert("end", text)

    def render_probability_chart(self, labels, probabilities):
        try:
            if self.chart_canvas is not None:
                self.chart_canvas.get_tk_widget().destroy()
                self.chart_canvas = None
            if self.chart_figure is not None:
                plt.close(self.chart_figure)
                self.chart_figure = None

            fig, ax = plt.subplots(figsize=(5.4, 2.7), dpi=100)
            colors_list = ["#0F4C81", "#0EA5E9", "#9333EA", "#16A34A", "#F59E0B", "#DC2626"]
            bar_colors = [colors_list[i % len(colors_list)] for i in range(len(labels))]
            bars = ax.bar(labels, probabilities, color=bar_colors)

            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Class Probability Distribution", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.25)

            for bar, value in zip(bars, probabilities):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f"{value:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()

            self.chart_figure = fig
            self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_host)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as exc:
            write_error_log("render_probability_chart", exc)

    def build_report_text(self, prediction, confidence, score, risk, answers, probability_map):
        lines = [
            "AI CLINICAL SCREENING REPORT",
            "=" * 78,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Patient Name: {self.patient_name.get().strip() or 'Not provided'}",
            f"Patient ID: {self.patient_id.get().strip() or 'Not provided'}",
            f"Age: {self.patient_age.get().strip() or 'Not provided'}",
            f"Gender: {self.patient_gender.get().strip() or 'Not provided'}",
            f"Clinic / Department: {self.clinic_name.get().strip() or 'Not provided'}",
            "",
            "CLINICAL SUMMARY",
            "-" * 78,
            f"Predicted Class: {prediction}",
            f"Model Confidence: {confidence:.2%}",
            f"Symptom Score: {score} / 24",
            f"Risk Level: {risk}",
            "",
            "PROBABILITY DISTRIBUTION",
            "-" * 78,
        ]
        for label, prob in probability_map.items():
            lines.append(f"{label}: {prob:.2%}")

        lines.extend(
            [
                "",
                "QUESTIONNAIRE RESPONSES",
                "-" * 78,
            ]
        )
        for idx, feature in enumerate(FEATURES, start=1):
            value = answers[feature]
            lines.append(f"{idx:02d}. {feature}: {REVERSE_ANSWER_MAP[value]} ({value})")

        lines.extend(
            [
                "",
                "AI INTERPRETATION",
                "-" * 78,
                "This result is intended to support early screening workflow, triage, and structured intake.",
                "It should be interpreted with history, examination, imaging, laboratory findings, and physician judgment.",
                "",
                "COMMERCIAL VALUE FOR CLINIC",
                "-" * 78,
                "1. Faster triage and more organized intake workflow.",
                "2. Better documentation consistency.",
                "3. More professional patient-facing reporting.",
                "4. Better support for follow-up comparison and service quality improvement.",
                "",
                "MEDICAL DISCLAIMER",
                "-" * 78,
                "This software is an AI-assisted screening tool and does not provide a final diagnosis.",
            ]
        )
        return "\n".join(lines)

    def build_report_html(self, prediction, confidence, score, risk, answers, probability_map):
        risk_color = self.get_risk_color(risk)

        answer_rows = []
        for idx, feature in enumerate(FEATURES, start=1):
            answer_rows.append(
                f"<tr><td>{idx:02d}</td><td>{feature}</td><td>{REVERSE_ANSWER_MAP[answers[feature]]}</td></tr>"
            )

        prob_rows = []
        for label, prob in probability_map.items():
            prob_rows.append(f"<tr><td>{label}</td><td>{prob:.2%}</td></tr>")

        logo_html = ""
        if self.logo_path and os.path.exists(self.logo_path):
            logo_uri = self.logo_path.replace("\\", "/")
            logo_html = f'<img src="file:///{logo_uri}" style="max-height:70px;max-width:220px;">'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Clinical Screening Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 32px;
    color: #1f2937;
    background: #f8fbfe;
}}
.container {{
    max-width: 1000px;
    margin: auto;
    background: white;
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}
.header {{
    background: #0b3a63;
    color: white;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 24px;
}}
.brand {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}}
.badge {{
    display: inline-block;
    background: #e8f3ff;
    color: #0b3a63;
    padding: 8px 12px;
    border-radius: 999px;
    font-weight: bold;
    font-size: 12px;
}}
.cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}}
.card {{
    background: #f8fbfe;
    border: 1px solid #d7e3f0;
    border-radius: 14px;
    padding: 16px;
}}
.card h3 {{
    margin: 0 0 8px 0;
    font-size: 13px;
    color: #64748b;
}}
.card p {{
    margin: 0;
    font-size: 24px;
    font-weight: bold;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    margin-bottom: 24px;
}}
th, td {{
    border: 1px solid #dbe4ee;
    padding: 10px;
    text-align: left;
}}
th {{
    background: #eef5fb;
}}
.section-title {{
    font-size: 20px;
    color: #123b5d;
    margin-top: 20px;
    margin-bottom: 8px;
}}
.disclaimer {{
    background: #fff7ed;
    color: #7c2d12;
    border: 1px solid #f6d7b0;
    border-radius: 12px;
    padding: 16px;
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="brand">
            <div>
                <div class="badge">CLINICAL AI SUPPORT</div>
                <h1>AI Joint Disease Screening Report</h1>
                <p>Professional questionnaire-based clinical decision support</p>
            </div>
            <div>{logo_html}</div>
        </div>
    </div>

    <div class="cards">
        <div class="card"><h3>Prediction</h3><p style="color:#0F4C81;">{prediction}</p></div>
        <div class="card"><h3>Confidence</h3><p style="color:#0E7490;">{confidence:.1%}</p></div>
        <div class="card"><h3>Symptom Score</h3><p style="color:#9333EA;">{score}</p></div>
        <div class="card"><h3>Risk Level</h3><p style="color:{risk_color};">{risk}</p></div>
    </div>

    <div class="section-title">Patient Information</div>
    <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Date</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        <tr><td>Patient Name</td><td>{self.patient_name.get().strip() or 'Not provided'}</td></tr>
        <tr><td>Patient ID</td><td>{self.patient_id.get().strip() or 'Not provided'}</td></tr>
        <tr><td>Age</td><td>{self.patient_age.get().strip() or 'Not provided'}</td></tr>
        <tr><td>Gender</td><td>{self.patient_gender.get().strip() or 'Not provided'}</td></tr>
        <tr><td>Clinic / Department</td><td>{self.clinic_name.get().strip() or 'Not provided'}</td></tr>
    </table>

    <div class="section-title">Probability Distribution</div>
    <table>
        <tr><th>Class</th><th>Probability</th></tr>
        {''.join(prob_rows)}
    </table>

    <div class="section-title">Questionnaire Responses</div>
    <table>
        <tr><th>#</th><th>Feature</th><th>Answer</th></tr>
        {''.join(answer_rows)}
    </table>

    <div class="section-title">Commercial Value for Clinic</div>
    <p>Faster triage, more structured documentation, professional reporting, and better follow-up comparison.</p>

    <div class="disclaimer">
        <strong>Medical Disclaimer:</strong> This software is an AI-assisted screening tool and does not provide a final diagnosis.
    </div>
</div>
</body>
</html>
"""

    def analyze_patient(self):
        try:
            self.validate_age()
            answers = self.collect_answers()

            ordered = []
            for feature in self.feature_columns:
                if feature not in answers:
                    raise ValueError(f"Missing feature in questionnaire: {feature}")
                ordered.append(answers[feature])

            df = pd.DataFrame([ordered], columns=self.feature_columns)
            pred = self.model.predict(df)[0]

            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(df)[0]
            else:
                prob = [1.0]

            prediction = self.get_prediction_display(pred)
            labels = self.get_class_labels()
            if len(labels) != len(prob):
                labels = [f"Class {i + 1}" for i in range(len(prob))]

            probability_map = {label: float(p) for label, p in zip(labels, prob)}
            top_confidence = max(probability_map.values())
            score = self.calculate_symptom_score(answers)
            risk = self.calculate_risk_level(top_confidence, score)

            self.last_result_text = self.build_report_text(
                prediction, top_confidence, score, risk, answers, probability_map
            )
            self.last_html_report = self.build_report_html(
                prediction, top_confidence, score, risk, answers, probability_map
            )
            self.last_prediction = prediction
            self.last_confidence = top_confidence
            self.last_answers = answers
            self.last_probabilities = list(probability_map.values())
            self.last_probability_labels = list(probability_map.keys())
            self.last_symptom_score = score
            self.last_risk_level = risk

            self.set_result_text(self.last_result_text)
            self.update_dashboard(prediction, top_confidence, score, risk)
            self.render_probability_chart(self.last_probability_labels, self.last_probabilities)

            messagebox.showinfo("Analysis Completed", "Patient analysis completed successfully.")
        except Exception as exc:
            write_error_log("analyze_patient", exc)
            messagebox.showerror("Analysis Error", str(exc))

    def clear_results_only(self):
        self.set_result_text("Results cleared. Ready for a new analysis.")
        self.clear_dashboard()
        self.render_probability_chart(["No Data"], [0.0])
        messagebox.showinfo("Cleared", "Only results and dashboard have been cleared.")

    def reset_full_form(self):
        self.patient_name.delete(0, "end")
        self.patient_age.delete(0, "end")
        self.patient_id.delete(0, "end")
        self.clinic_name.delete(0, "end")
        self.patient_gender.set("Female")

        for feature in FEATURES:
            self.answer_boxes[feature].set("I don't know")

        self.last_result_text = ""
        self.last_html_report = ""
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_answers = None
        self.last_probabilities = []
        self.last_probability_labels = []
        self.last_symptom_score = 0
        self.last_risk_level = "Not calculated"

        self.set_result_text("Form reset completed.")
        self.clear_dashboard()
        self.render_probability_chart(["No Data"], [0.0])
        messagebox.showinfo("Reset", "Full form has been reset successfully.")

    def preview_current_answers(self):
        try:
            self.validate_age()
            answers = self.collect_answers()

            lines = [
                "CURRENT ANSWERS PREVIEW",
                "=" * 78,
                f"Patient Name: {self.patient_name.get().strip() or 'Not provided'}",
                f"Patient ID: {self.patient_id.get().strip() or 'Not provided'}",
                f"Age: {self.patient_age.get().strip() or 'Not provided'}",
                f"Gender: {self.patient_gender.get().strip() or 'Not provided'}",
                f"Clinic / Department: {self.clinic_name.get().strip() or 'Not provided'}",
                "",
            ]
            for idx, feature in enumerate(FEATURES, start=1):
                value = answers[feature]
                lines.append(f"{idx:02d}. {feature}: {REVERSE_ANSWER_MAP[value]} ({value})")

            self.set_result_text("\n".join(lines))
        except Exception as exc:
            write_error_log("preview_current_answers", exc)
            messagebox.showerror("Preview Error", str(exc))

    def show_last_result_again(self):
        if not self.last_result_text:
            messagebox.showwarning("No Result", "There is no previous result to display.")
            return

        self.set_result_text(self.last_result_text)
        self.update_dashboard(
            self.last_prediction,
            self.last_confidence,
            self.last_symptom_score,
            self.last_risk_level,
        )
        if self.last_probability_labels and self.last_probabilities:
            self.render_probability_chart(self.last_probability_labels, self.last_probabilities)
        messagebox.showinfo("Last Result", "Previous result displayed again.")

    def save_patient_record_to_excel(self):
        try:
            if not self.last_answers or not self.last_prediction:
                messagebox.showwarning("No Analysis", "Please analyze patient first.")
                return

            excel_path = make_excel_filename()
            row = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Patient Name": self.patient_name.get().strip() or "Not provided",
                "Patient ID": self.patient_id.get().strip() or "Not provided",
                "Age": self.patient_age.get().strip() or "Not provided",
                "Gender": self.patient_gender.get().strip() or "Not provided",
                "Clinic / Department": self.clinic_name.get().strip() or "Not provided",
                "Prediction": self.last_prediction,
                "Confidence (%)": round(self.last_confidence * 100, 2),
                "Symptom Score": self.last_symptom_score,
                "Risk Level": self.last_risk_level,
            }
            for feature in FEATURES:
                row[feature] = REVERSE_ANSWER_MAP[self.last_answers[feature]]

            new_df = pd.DataFrame([row])
            if os.path.exists(excel_path):
                old_df = pd.read_excel(excel_path)
                final_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            final_df.to_excel(excel_path, index=False)
            messagebox.showinfo("Saved", f"Patient record saved successfully:\n{excel_path}")
        except Exception as exc:
            write_error_log("save_patient_record_to_excel", exc)
            messagebox.showerror("Excel Save Error", str(exc))

    def export_txt_report(self):
        try:
            if not self.last_result_text:
                messagebox.showwarning("No Report", "There is no report to export.")
                return

            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                initialfile=os.path.basename(make_report_filename("txt")),
                initialdir=get_app_records_directory(),
            )
            if not filename:
                return

            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.last_result_text)

            messagebox.showinfo("Exported", f"TXT report exported successfully:\n{filename}")
        except Exception as exc:
            write_error_log("export_txt_report", exc)
            messagebox.showerror("TXT Export Error", str(exc))

    def export_html_report(self):
        try:
            if not self.last_html_report:
                messagebox.showwarning("No Report", "There is no report to export.")
                return

            filename = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")],
                initialfile=os.path.basename(make_report_filename("html")),
                initialdir=get_app_records_directory(),
            )
            if not filename:
                return

            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.last_html_report)

            if messagebox.askyesno("Exported", "HTML report exported successfully.\nOpen it now?"):
                webbrowser.open_new_tab(filename)
        except Exception as exc:
            write_error_log("export_html_report", exc)
            messagebox.showerror("HTML Export Error", str(exc))

    def export_pdf_report(self):
        try:
            if not REPORTLAB_AVAILABLE:
                messagebox.showerror(
                    "PDF Export Error",
                    "ReportLab is not installed.\nRun: pip install reportlab pillow",
                )
                return

            if not self.last_result_text or not self.last_answers:
                messagebox.showwarning("No Report", "There is no report to export.")
                return

            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
                initialfile=os.path.basename(make_report_filename("pdf")),
                initialdir=get_app_records_directory(),
            )
            if not filename:
                return

            doc = SimpleDocTemplate(
                filename,
                pagesize=A4,
                rightMargin=1.5 * cm,
                leftMargin=1.5 * cm,
                topMargin=1.4 * cm,
                bottomMargin=1.4 * cm,
            )

            styles = getSampleStyleSheet()
            title_style = styles["Title"]
            heading_style = styles["Heading2"]
            normal = styles["BodyText"]
            normal.spaceAfter = 6
            small = ParagraphStyle(
                "small",
                parent=styles["BodyText"],
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#7C2D12"),
            )

            story = []

            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    story.append(RLImage(self.logo_path, width=4.5 * cm, height=1.6 * cm))
                    story.append(Spacer(1, 0.2 * cm))
                except Exception:
                    pass

            story.append(Paragraph("AI Joint Disease Screening Report", title_style))
            story.append(Paragraph("Clinical AI Support", heading_style))
            story.append(Spacer(1, 0.2 * cm))

            patient_data = [
                ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Patient Name", self.patient_name.get().strip() or "Not provided"],
                ["Patient ID", self.patient_id.get().strip() or "Not provided"],
                ["Age", self.patient_age.get().strip() or "Not provided"],
                ["Gender", self.patient_gender.get().strip() or "Not provided"],
                ["Clinic / Department", self.clinic_name.get().strip() or "Not provided"],
            ]
            patient_table = Table(patient_data, colWidths=[5 * cm, 10.5 * cm])
            patient_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5FB")),
                    ]
                )
            )
            story.append(patient_table)
            story.append(Spacer(1, 0.35 * cm))

            summary_data = [
                ["Prediction", self.last_prediction],
                ["Confidence", f"{self.last_confidence:.2%}"],
                ["Symptom Score", str(self.last_symptom_score)],
                ["Risk Level", self.last_risk_level],
            ]
            summary_table = Table(summary_data, colWidths=[5 * cm, 10.5 * cm])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5FB")),
                    ]
                )
            )
            story.append(summary_table)
            story.append(Spacer(1, 0.35 * cm))

            story.append(Paragraph("Probability Distribution", heading_style))
            prob_rows = [["Class", "Probability"]]
            for label, prob in zip(self.last_probability_labels, self.last_probabilities):
                prob_rows.append([label, f"{prob:.2%}"])
            prob_table = Table(prob_rows, colWidths=[8 * cm, 7.5 * cm])
            prob_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEEFF")),
                    ]
                )
            )
            story.append(prob_table)
            story.append(Spacer(1, 0.35 * cm))

            story.append(Paragraph("Questionnaire Responses", heading_style))
            answer_rows = [["#", "Feature", "Answer"]]
            for idx, feature in enumerate(FEATURES, start=1):
                answer_rows.append(
                    [
                        str(idx),
                        feature,
                        REVERSE_ANSWER_MAP[self.last_answers[feature]],
                    ]
                )
            answer_table = Table(answer_rows, colWidths=[1.2 * cm, 10.2 * cm, 4.1 * cm])
            answer_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEEFF")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(answer_table)
            story.append(Spacer(1, 0.35 * cm))

            story.append(Paragraph("Commercial Value for Clinic", heading_style))
            story.append(
                Paragraph(
                    "Faster triage, improved intake consistency, professional reporting, and stronger follow-up documentation.",
                    normal,
                )
            )
            story.append(Spacer(1, 0.2 * cm))
            story.append(
                Paragraph(
                    "Medical Disclaimer: This software is an AI-assisted screening tool and does not provide a final diagnosis.",
                    small,
                )
            )

            doc.build(story)
            if messagebox.askyesno("PDF Exported", "PDF report exported successfully.\nOpen it now?"):
                open_with_default_app(filename)
        except Exception as exc:
            write_error_log("export_pdf_report", exc)
            messagebox.showerror("PDF Export Error", str(exc))

    def open_records_folder(self):
        try:
            open_with_default_app(get_app_records_directory())
        except Exception as exc:
            write_error_log("open_records_folder", exc)
            messagebox.showerror("Open Folder Error", str(exc))

    def print_last_report(self):
        try:
            if not self.last_result_text:
                messagebox.showwarning("No Report", "There is no report to print.")
                return

            tmp_path = os.path.join(tempfile.gettempdir(), f"joint_report_{make_timestamp()}.txt")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(self.last_result_text)

            if sys.platform.startswith("win"):
                os.startfile(tmp_path, "print")
            else:
                open_with_default_app(tmp_path)

            messagebox.showinfo("Print", "The report has been sent to the default printing flow.")
        except Exception as exc:
            write_error_log("print_last_report", exc)
            messagebox.showerror("Print Error", str(exc))


def main():
    try:
        app = DiagnosisApp()
        app.mainloop()
    except Exception as exc:
        write_error_log("main", exc)
        print("Application crashed:")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
