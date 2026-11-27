import logging
import math
from dash import Dash, html, dcc, Input, Output, State, ALL, MATCH, ctx
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from pipeline.visionmanager import VisionManager, pin_current_thread_little_cores
import time
from threading import Thread
from util.config import ConfigCategory
from util.system_stats import get_system_stats
from camera.camerareader import RESOLUTIONS
from localization.undistort import has_opencv_calibration
from calibration.charuco_calibrator import save_samples
import os
import sys

logging.getLogger("werkzeug").setLevel(logging.WARNING)


DASHBOARD_BASE_PORT = 5800

# match to v4l2-ctl --list-ctrls
EXPOSURE_SLIDER_MAX = 100  # v4l2 units of 0.1ms
GAIN_SLIDER_MAX = 255

STREAM_MAX_FPS = 15
STREAM_MAX_WIDTH = 640


_PREPROCESS_SLIDER_SPECS = {
    "dfc-target": ("target_brightness", 155, True, "SET_TARGET_BRIGHTNESS"),
    "aslc-bins": ("acc_num_bins", 1200, True, "SET_NUM_BINS"),
    "aslc-min-corr": ("min_corr_strength", 0.1, False, "SET_MIN_CORR_STRENGTH"),
    "divergence-gain": ("divergence_gain", 1.5, False, "SET_DIVERGENCE_GAIN"),
}


def _preprocess_cfg(slider_type: str):
    key, default, is_int, _ = _PREPROCESS_SLIDER_SPECS[slider_type]
    cat = ConfigCategory("Preprocessing")
    return cat.getIntConfig(key, default) if is_int else cat.getFloatConfig(key, default)


def _preprocess_setter(slider_type: str):
    _, _, is_int, fn_name = _PREPROCESS_SLIDER_SPECS[slider_type]
    cfg = _preprocess_cfg(slider_type)

    def _set(value):
        val = int(value) if is_int else float(value)
        mod = sys.modules.get("camera.preprocess")
        if mod is not None:
            getattr(mod, fn_name)(val)
        elif is_int:
            cfg.setInt(val)
        else:
            cfg.setFloat(val)

    return _set


def dashboard_port(camera_id: int) -> int:
    return DASHBOARD_BASE_PORT + camera_id


MJPEG_PART_HEADER = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'


def mjpeg_part(jpeg: bytes) -> bytes:
    return jpeg + b'\r\n' + MJPEG_PART_HEADER


def _build_blank_jpeg(text: str, text_pos: tuple, quality: int = 60) -> bytes:
    blank = 255 * np.ones((480, 640, 3), dtype="uint8")
    cv2.putText(blank, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    ok, buffer = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else b""


def _stream_mjpeg_frames(vision_manager: VisionManager, camera_id: int, quality_fn, unavailable_text: str, text_pos: tuple):
    blank_frame_cache = None
    last_seq = None
    next_send = 0.0
    frame_interval = 1.0 / STREAM_MAX_FPS

    yield MJPEG_PART_HEADER
    while True:
        pipeline = vision_manager.get_pipeline(camera_id)
        if not pipeline or not pipeline.is_enabled():
            if blank_frame_cache is None:
                blank_frame_cache = _build_blank_jpeg(unavailable_text, text_pos, quality=60)
            last_seq = None
            yield mjpeg_part(blank_frame_cache)
            time.sleep(0.5)
            continue

        wait_s = next_send - time.monotonic()
        if wait_s > 0:
            time.sleep(wait_s)

        seq = pipeline.get_frame_seq() if hasattr(pipeline, "get_frame_seq") else None
        if seq is not None and seq == last_seq:
            time.sleep(0.01)
            continue

        quality = quality_fn()
        source, jpeg = pipeline.get_preview_jpeg(quality, STREAM_MAX_WIDTH)
        if source is None or jpeg is None:
            time.sleep(0.01)
            continue

        if seq is None:
            if source is last_seq:
                time.sleep(0.01)
                continue
            last_seq = source
        else:
            last_seq = seq

        next_send = time.monotonic() + frame_interval
        yield mjpeg_part(jpeg)


class DashboardServer:
    config_category = ConfigCategory("HTMLServer")
    framecomp_slider = config_category.getFloatConfig("framecomp_slider", 0.5)

    def __init__(self, vision_manager: VisionManager, camera_id: int):
        self.vision_manager = vision_manager
        self.camera_id = camera_id
        self.server = Flask(__name__)
        self.app = Dash(__name__, server=self.server, suppress_callback_exceptions=True)
        self.app.index_string = self.index_string()
        
        self._calibrator = None
        
        self.setup_layout()
        self.setup_callbacks()
        self.start_server_thread()

    @property
    def calibrator(self):
        if self._calibrator is None:
            from calibration.charuco_calibrator import CharucoCalibrator
            pipeline = self.vision_manager.get_pipeline(self.camera_id)
            res = pipeline.get_resolution() if pipeline is not None else (1280, 800)
            cb = pipeline.update_optical_health if hasattr(pipeline, "update_optical_health") else None
            self._calibrator = CharucoCalibrator(self.camera_id, resolution=res, on_health_update=cb)
        return self._calibrator

    def create_detection_content(self, cam_id):
        pipeline = self.vision_manager.get_pipeline(cam_id)
        
        if not pipeline:
            return html.Div(f"Camera {cam_id} not available")
        
        pipeline_type = pipeline.get_pipeline_type()
        if pipeline_type == "apriltag":
            settings_panel = self.create_apriltag_sliders(cam_id)
            side_title = "Detections"
        else:
            settings_panel = self.create_gamepiece_sliders(cam_id)
            side_title = "Detections"
        
        return html.Div([
                html.Div([
                    html.Div([
                        html.H4(side_title, style={
                            'textAlign': 'left',
                            'color': '#CCC9CA',
                            'font-size': '18px',
                            'font-weight': 'medium',
                            'padding': '0px 0px 0px 7px',
                        }),
                        html.Div(
                            id=f'detections-{cam_id}',
                            style={
                                'flex-direction': 'column',
                                'gap': '5px',
                                'margin-top': '5px',
                                'padding': '0 0px',
                                'max-height': '310px',
                                'overflow-y': 'auto',
                                'overflow-x': 'hidden',
                                'flex-shrink': '0',
                            }
                        ),
                        html.Br(),
                        settings_panel,
                        self.create_exposure_controls(cam_id),
                    ], style={
                        "flex": "1",
                        "padding": "10px",
                        "color": "#FFF",
                        "flex-direction": "column",
                        "display": "flex",
                        "flex-grow": "1",
                        "min-width": "40%",
                        "max-width": "50%",
                        "box-sizing": "border-box",
                        "overflow-y": "auto",
                        "overflow-x": "hidden",
                        "height": "85vh",
                    }),
                    
                    html.Div([
                        html.Div([
                            html.Div(f"{pipeline.get_camera_name()}", style={
                                'color': '#CDA646',
                                'font-size': '12px',
                                'font-weight': 'bold',
                                'margin-bottom': '5px',
                                'text-align': 'center',
                            }),
                            html.Img(
                                src=f"/video_feed/{cam_id}",
                                style={
                                    "width": "100%",
                                    "max-width": "650px",
                                    "max-height": "600px",
                                    "border": "3px solid #CDA646",
                                    "border-radius": "9px",
                                }
                            ),
                            html.Div(id=f'metrics-{cam_id}', style={
                                "position": "relative",
                                "width": "100%",
                                "max-width": "600px",
                                "height": "40px",
                                "margin-top": "5px",
                            }),
                            html.Div(id=f'system-{cam_id}', style={
                                "position": "relative",
                                "width": "100%",
                                "max-width": "600px",
                                "height": "32px",
                            }),
                            html.Br(),
                            html.Div([
                                html.Button("Reload", id={'type': 'reload', 'index': cam_id}, style={
                                    "margin": "10px 5px",
                                    "font-size": "14px",
                                    "color": "#161616",
                                    "background-color": "rgba(100, 200, 255, 1)",
                                    "border": "none",
                                    "padding": "8px 16px",
                                    "width": "145px",
                                    "height": "40px",
                                    "border-radius": "20px",
                                    "cursor": "pointer",
                                    "font-weight": "bold"
                                }),
                                html.Button("Reboot", id={'type': 'reboot', 'index': cam_id}, style={
                                    "margin": "10px 5px",
                                    "font-size": "14px",
                                    "color": "#161616",
                                    "background-color": "rgba(255, 204, 74, 1)",
                                    "border": "none",
                                    "padding": "8px 16px",
                                    "width": "145px",
                                    "height": "40px",
                                    "border-radius": "20px",
                                    "cursor": "pointer",
                                    "font-weight": "bold"
                                }),
                            ], style={
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "padding": "10px",
                            }),
                        ], style={
                            "display": "flex",
                            "flex-direction": "column",
                            "align-items": "center",
                            "justify-content": "center",
                        }),
                    ], style={
                        "flex": "3",
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "padding": "20px",
                        "height": "85vh",
                    })
                ], style={
                    "display": "flex",
                    "flex-direction": "row",
                    "width": "100%",
                    "padding": "10px 10px",
                    "overflow-x": "hidden",
                })
            ])

    def create_camera_switcher(self):
        camera_buttons = []
        for cam_id, pipeline in sorted(self.vision_manager.get_all_pipelines().items()):
            is_current = cam_id == self.camera_id
            label = f"{pipeline.get_camera_name()} ({cam_id})"
            if not pipeline.is_enabled():
                label += " · off"
            style = {
                "display": "inline-block",
                "margin": "0 8px 8px 0",
                "padding": "8px 14px",
                "border-radius": "16px",
                "font-size": "13px",
                "font-weight": "bold",
                "text-decoration": "none",
                "color": "#161616" if is_current else "#CCC9CA",
                "background-color": "#CDA646" if is_current else "rgba(255,255,255,0.08)",
                "border": "1px solid #CDA646" if is_current else "1px solid rgba(255,255,255,0.2)",
                "cursor": "pointer",
            }
            if is_current:
                camera_buttons.append(html.Span(label, style=style))
            else:
                camera_buttons.append(html.A(
                    label,
                    href="#",
                    className="camera-nav-link",
                    **{"data-port": str(dashboard_port(cam_id))},
                    style=style,
                ))

        return html.Div([
            html.Div("Cameras", style={
                "color": "#CCC9CA",
                "font-size": "12px",
                "font-weight": "bold",
                "margin-bottom": "8px",
                "letter-spacing": "0.04em",
                "text-transform": "uppercase",
            }),
            html.Div([
                html.A(
                    "All Cameras",
                    href="#",
                    className="camera-hub-link",
                    **{"data-port": str(DASHBOARD_BASE_PORT)},
                    style={
                        "display": "inline-block",
                        "margin": "0 8px 8px 0",
                        "padding": "8px 14px",
                        "border-radius": "16px",
                        "font-size": "13px",
                        "font-weight": "bold",
                        "text-decoration": "none",
                        "color": "#161616",
                        "background-color": "rgba(100, 200, 255, 1)",
                    },
                ),
                *camera_buttons,
            ]),
        ], style={
            "margin": "0 25px 15px 25px",
            "padding": "12px 0 4px 0",
            "border-bottom": "1px solid rgba(255,255,255,0.1)",
        })

    def setup_layout(self):
        camera_content = self.create_detection_content(self.camera_id)
        
        self.app.layout = html.Div([
            html.Div([
                html.H1("MonkeyVision", style={
                    'textAlign': 'left',
                    'color': '#CCC9CA',
                    'font-size': '32px',
                    'font-weight': 'bold',
                    'padding': '10px 20px 0 25px',
                    'margin-bottom': '5px',
                }),
                html.H4("By Team 846 • The Funky Monkeys", style={
                    'textAlign': 'left',
                    'color': '#CCC9CA',
                    'font-size': '14px',
                    'font-weight': 'regular',
                    'margin': '5px 0 10px 25px',
                }),
                html.Img(
                    src="/assets/logo.svg",
                    style={
                        "position": "absolute",
                        "top": "20px",
                        "right": "20px",
                        "width": "50px",
                        "height": "50px"
                    }
                ),
            ]),

            self.create_camera_switcher(),
            
            dcc.Tabs(id='tabs', value='detection', children=[
                dcc.Tab(label='Detection', value='detection', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Calibration', value='calibration', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Focus', value='focus', className='custom-tab', selected_className='custom-tab--selected'),
            ], style={
                'margin': '0 25px',
                'border-bottom': '2px solid #CDA646',
            }),
            
            html.Div(id='tab-content', children=camera_content),
            
            dcc.Interval(id="update-interval", interval=500, n_intervals=0),  # 500ms for faster progress updates
        ], style={
            "background-color": "#161616",
            "color": "#FFF",
            "font-family": "'Inter', sans-serif",
            "min-height": "100vh",
            "padding": "0",
            "margin": "0",
        })

    def create_exposure_controls(self, cam_id):
        pipeline = self.vision_manager.get_pipeline(cam_id)
        if not pipeline:
            return html.Div()
        settings = pipeline.cam.get_exposure_settings()
        auto = settings["auto_exposure"]

        resolution_controls = []
        if pipeline.cam.use_preprocessing:
            width, height = pipeline.cam.get_resolution()
            resolution_controls = [
                html.Label("Resolution", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    'padding': '10px 20px 0 15px',
                }),
                dcc.RadioItems(
                    id={'type': 'resolution', 'index': cam_id},
                    options=[{'label': f' {w}x{h}', 'value': f'{w}x{h}'} for w, h in RESOLUTIONS],
                    value=f'{width}x{height}',
                    inline=True,
                    labelStyle={'margin-right': '20px'},
                    style={
                        "color": "#CCC9CA",
                        "font-size": "16px",
                        'padding': '10px 20px 0 15px',
                    },
                ),
                html.Div(self._resolution_status(cam_id, width, height), id={'type': 'resolution-status', 'index': cam_id}, style={
                    'color': '#CCC9CA',
                    'font-size': '13px',
                    'padding': '5px 20px 0 15px',
                }),
            ]

        return html.Div([
            html.H4("Camera", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),

            *resolution_controls,

            dcc.Checklist(
                id={'type': 'auto-exposure', 'index': cam_id},
                options=[{'label': ' Auto Gain', 'value': 'auto'}],
                value=['auto'] if auto else [],
                style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    'padding': '10px 20px 15px 15px',
                },
            ),

            html.Label("Exposure (x100 µs)", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'exposure', 'index': cam_id},
                min=1,
                max=EXPOSURE_SLIDER_MAX,
                step=1,
                value=min(settings["exposure"], EXPOSURE_SLIDER_MAX),
                marks={1: '1', EXPOSURE_SLIDER_MAX: str(EXPOSURE_SLIDER_MAX)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),

            html.Label("Gain", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'gain', 'index': cam_id},
                min=0,
                max=GAIN_SLIDER_MAX,
                step=1,
                value=min(settings["gain"], GAIN_SLIDER_MAX),
                disabled=auto,
                marks={0: '0', GAIN_SLIDER_MAX: str(GAIN_SLIDER_MAX)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),

            html.Button("Retune Gain", id={'type': 'retune-gain', 'index': cam_id}, disabled=not auto, style={
                "margin": "10px 15px 0",
                "font-size": "14px",
                "color": "#161616",
                "background-color": "rgba(100, 200, 255, 1)",
                "border": "none",
                "padding": "8px 16px",
                "width": "145px",
                "height": "40px",
                "border-radius": "20px",
                "cursor": "pointer",
                "font-weight": "bold"
            }),

            html.Div(settings["status"], id={'type': 'exposure-status', 'index': cam_id}, style={
                'color': '#CCC9CA',
                'font-size': '13px',
                'padding': '15px 20px 0 15px',
            }),
        ])

    def _resolution_status(self, cam_id, width, height):
        if has_opencv_calibration(cam_id, width, height):
            return f"Using {width}x{height} calibration"
        return f"No calibration at {width}x{height}: tags won't report distance until you calibrate on the Calibration tab"

    def create_apriltag_sliders(self, cam_id):
        from localization.detection import GET_THRESH_STEP, GET_THRESH_WIN
        target_brightness = _preprocess_cfg("dfc-target").valueInt()
        num_bins = _preprocess_cfg("aslc-bins").valueInt()
        min_corr = _preprocess_cfg("aslc-min-corr").valueFloat()
        div_gain = _preprocess_cfg("divergence-gain").valueFloat()
        
        return html.Div([
            html.H4("Settings", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            
            html.Label("Dynamic Frame Correction Target", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                "margin-bottom": "10px",
                'padding': '15px 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'dfc-target', 'index': cam_id},
                min=50,
                max=200,
                step=1,
                value=target_brightness,
                marks={50: '50', 200: '200'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("ASLC Num Bins", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'aslc-bins', 'index': cam_id},
                min=100,
                max=1600,
                step=100,
                value=num_bins,
                marks={100: '100', 1600: '1600'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Min ASLC Correction", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'aslc-min-corr', 'index': cam_id},
                min=0.02,
                max=0.5,
                step=0.01,
                value=min_corr,
                marks={0.02: '0.02', 0.5: '0.5'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Divergence Gain", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'divergence-gain', 'index': cam_id},
                min=0.5,
                max=4.0,
                step=0.1,
                value=div_gain,
                marks={0.5: '0.5', 4.0: '4.0'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Thresholding Steps", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'thresh-step', 'index': cam_id},
                min=11,
                max=33,
                step=2,
                value=GET_THRESH_STEP(),
                marks={11: '11', 33: '33'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Maximum Threshold Window", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'max-thresh', 'index': cam_id},
                min=11,
                max=33,
                step=2,
                value=GET_THRESH_WIN(),
                marks={11: '11', 33: '33'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Displayed Frame Quality", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'framecomp', 'index': cam_id},
                min=0.05,
                max=1,
                step=0.05,
                value=DashboardServer.framecomp_slider.valueFloat(),
                marks={0.05: '5%', 1: '100%'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
        ])

    def create_calibration_tab(self, cam_id):
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Camera Calibration", style={
                        'textAlign': 'left',
                        'color': '#CCC9CA',
                        'font-size': '18px',
                        'font-weight': 'medium',
                        'padding': '0px 0px 0px 7px',
                    }),
                    
                    html.Div([
                        html.P("https://calib.io/pages/camera-calibration-pattern-generator", style={'margin': '10px 0'}),
                        html.P("11 rows, 8 columns", style={'margin': '10px 0'}),
                        html.P("15mm ChArUco checkers. Class 4x4.", style={'margin': '10px 0'}),
                        html.P("Printed to 21.2mm checkers, 15.55mm markers", style={'margin': '10px 0'}),
                    ], style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'padding': '15px 20px',
                        'border': '2px solid rgba(255, 255, 255, 0.3)',
                        'border-radius': '10px',
                        'margin': '15px 7px',
                    }),

                    dcc.Checklist(
                        id={'type': 'save-cal-samples', 'index': cam_id},
                        options=[{'label': ' Save sample images (to calibrate on another computer)', 'value': 'save'}],
                        value=['save'] if save_samples.valueInt() else [],
                        style={
                            "color": "#CCC9CA",
                            "font-size": "16px",
                            'padding': '0 20px 0 15px',
                        },
                    ),
                    
                    html.Div(id=f'calibration-status-{cam_id}', style={
                        'padding': '15px 20px',
                        'margin': '15px 7px',
                        'border': '2px solid rgba(255, 255, 255, 0.3)',
                        'border-radius': '10px',
                        'color': '#CCC9CA',
                        'font-size': '14px',
                    }),
                    
                    html.Div([
                        html.Button("Calibrate", id={'type': 'run-calibration', 'index': cam_id}, style={
                            "margin": "10px 5px",
                            "font-size": "14px",
                            "color": "#161616",
                            "background-color": "rgba(100, 200, 100, 1)",
                            "border": "none",
                            "padding": "8px 16px",
                            "width": "180px",
                            "height": "40px",
                            "border-radius": "20px",
                            "cursor": "pointer",
                            "font-weight": "bold"
                        }),
                        html.Button("Reset", id={'type': 'reset-calibration', 'index': cam_id}, style={
                            "margin": "10px 5px",
                            "font-size": "14px",
                            "color": "#161616",
                            "background-color": "rgba(255, 100, 100, 1)",
                            "border": "none",
                            "padding": "8px 16px",
                            "width": "180px",
                            "height": "40px",
                            "border-radius": "20px",
                            "cursor": "pointer",
                            "font-weight": "bold"
                        }),
                    ], style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "center",
                        "padding": "10px",
                    }),
                    
                ], style={
                    "flex": "1",
                    "padding": "10px",
                    "color": "#FFF",
                    "flex-direction": "column",
                    "display": "flex",
                    "flex-grow": "1",
                    "min-width": "40%",
                    "max-width": "50%",
                    "box-sizing": "border-box",
                    "overflow-y": "auto",
                    "overflow-x": "hidden",
                    "height": "85vh",
                }),
                
                html.Div([
                    html.Div([
                        html.Div(f"Calibration Preview - Camera {cam_id}", style={
                            'color': '#CDA646',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'margin-bottom': '5px',
                            'text-align': 'center',
                        }),
                        html.Img(
                            src=f"/calibration_feed/{cam_id}",
                            style={
                                "width": "100%",
                                "max-width": "650px",
                                "max-height": "600px",
                                "border": "3px solid #CDA646",
                            "border-radius": "9px",
                        }
                    ),
                ], style={
                    "display": "flex",
                    "flex-direction": "column",
                    "align-items": "center",
                    "justify-content": "center",
                }),
            ], style={
                "flex": "3",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "padding": "20px",
                "height": "85vh",
            })
        ], style={
            "display": "flex",
            "flex-direction": "row",
            "width": "100%",
            "padding": "10px 10px",
            "overflow-x": "hidden",
        }),
        
        html.Div(id=f'calibration-results-{cam_id}', style={
            "width": "100%",
            "padding": "20px",
            "display": "none",  # Hidden by default
        })
    ])

    def create_focus_tab(self, cam_id):
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Camera Focusing", style={
                        'textAlign': 'left',
                        'color': '#CCC9CA',
                        'font-size': '18px',
                        'font-weight': 'medium',
                        'padding': '0px 0px 0px 7px',
                    }),
                    html.P("Rotate the lens to maximize the score. Ensure the camera is pointed at a highly textured surface (like an AprilTag or calibration board).", style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'padding': '15px 20px',
                        'margin': '15px 7px',
                    }),
                    html.Div([
                        html.Button("Enable Focus Mode", id={'type': 'toggle-focus', 'index': cam_id}, style={
                            "margin": "10px 5px",
                            "font-size": "14px",
                            "color": "#161616",
                            "background-color": "rgba(255, 204, 74, 1)",
                            "border": "none",
                            "padding": "8px 16px",
                            "width": "200px",
                            "height": "40px",
                            "border-radius": "20px",
                            "cursor": "pointer",
                            "font-weight": "bold"
                        }),
                    ], style={
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "padding": "10px",
                    }),
                    html.Div(id=f'focus-score-{cam_id}', children="Score: 0.0", style={
                        'color': '#00ff00',
                        'font-size': '48px',
                        'font-weight': 'bold',
                        'text-align': 'center',
                        'padding': '20px',
                        'margin-top': '20px'
                    }),
                    html.P("use a longer exposure with low gain so sensor noise doesn't inflate the score. Switch back to match settings after", style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'padding': '0 20px',
                        'margin': '0 7px',
                    }),
                    self.create_exposure_controls(cam_id),
                ], style={
                    "flex": "1",
                    "padding": "10px",
                    "color": "#FFF",
                    "flex-direction": "column",
                    "display": "flex",
                    "flex-grow": "1",
                    "min-width": "40%",
                    "max-width": "50%",
                    "box-sizing": "border-box",
                    "overflow-y": "auto",
                    "overflow-x": "hidden",
                    "height": "85vh",
                }),
                html.Div([
                    html.Div([
                        html.Div(f"Focus Preview - Camera {cam_id}", style={
                            'color': '#CDA646',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'margin-bottom': '5px',
                            'text-align': 'center',
                        }),
                        html.Img(
                            src=f"/video_feed/{cam_id}",
                            style={
                                "width": "100%",
                                "max-width": "650px",
                                "max-height": "600px",
                                "border": "3px solid #CDA646",
                                "border-radius": "9px",
                            }
                        ),
                    ], style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "center",
                        "justify-content": "center",
                    }),
                ], style={
                    "flex": "3",
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                    "padding": "20px",
                    "height": "85vh",
                })
            ], style={
                "display": "flex",
                "flex-direction": "row",
                "width": "100%",
                "padding": "10px 10px",
                "overflow-x": "hidden",
            }),
        ])

    def create_gamepiece_sliders(self, cam_id):
        from localization.visiony import CONF, ASPECT_THRESH
        from localization.gamepiece_solution import WD, ONT
        
        return html.Div([
            html.H4("YOLO Settings", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            
            html.Label("Detection Confidence", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '15px 20px 0 15px'}),
            dcc.Slider(id={'type': 'yolo_conf', 'index': cam_id}, min=0.1, max=0.95, step=0.05, value=CONF.valueFloat(),
                      marks={0.1: '0.1', 0.5: '0.5', 0.95: '0.95'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.Label("Aspect Ratio Threshold", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '0 20px 0 15px'}),
            dcc.Slider(id={'type': 'yolo_aspect', 'index': cam_id}, min=0.0, max=1.0, step=0.05, value=ASPECT_THRESH.valueFloat(),
                      marks={0.0: '0.0', 0.5: '0.5', 1.0: '1.0'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.H4("Distance Settings", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            
            html.Label("Width Distance (WD)", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '15px 20px 0 15px'}),
            dcc.Slider(id={'type': 'wd', 'index': cam_id}, min=5.0, max=30.0, step=0.5, value=WD.valueFloat(),
                      marks={5.0: '5.0', 15.5: '15.5', 30.0: '30.0'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.Label("On Top Threshold (ONT)", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '0 20px 0 15px'}),
            dcc.Slider(id={'type': 'ont', 'index': cam_id}, min=0.0, max=20.0, step=1.0, value=ONT.valueFloat(),
                      marks={0.0: '0.0', 10.0: '10.0', 20.0: '20.0'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.Label("Displayed Frame Quality", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '0 20px 0 15px'}),
            dcc.Slider(id={'type': 'framecomp', 'index': cam_id}, min=0.05, max=1, step=0.05,
                      value=DashboardServer.framecomp_slider.valueFloat(),
                      marks={0.05: '5%', 1: '100%'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
        ])

    def setup_callbacks(self):
        from dash import no_update
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('tabs', 'value')]
        )
        def render_tab_content(tab):
            if tab == 'calibration':
                return self.create_calibration_tab(self.camera_id)
            elif tab == 'focus':
                return self.create_focus_tab(self.camera_id)
            else:  # detection tab
                return self.create_detection_content(self.camera_id)
        
        cam_id = self.camera_id

        @self.app.callback(
            [
                Output(f'metrics-{cam_id}', 'children'),
                Output(f'system-{cam_id}', 'children'),
                Output(f'detections-{cam_id}', 'children'),
            ],
            [Input('update-interval', 'n_intervals')]
        )
        def update_dashboard_panel(n_intervals, camera_id=cam_id):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if pipeline and pipeline.is_enabled():
                backend_fps = pipeline.get_backend_fps()
                fps_val = backend_fps if backend_fps > 0 else pipeline.get_framerate()
                cap_ms = pipeline.get_capture_latency() * 1000 if hasattr(pipeline, "get_capture_latency") else 0.0
                metrics_children = [
                    html.Span(f"Processing: {fps_val:.1f} FPS", style={
                        "position": "absolute",
                        "left": "0",
                        "bottom": "0",
                        "color": "rgba(255, 255, 255, 0.8)",
                        "font-size": "18px",
                        "font-weight": "regular",
                        "font-style": "italic",
                        "padding": "5px 10px",
                    }),
                    html.Span(
                        f"Latency: {pipeline.get_processing_latency() * 1000:.2f} ms",
                        title=f"Since capture: {cap_ms:.2f} ms",
                        style={
                            "position": "absolute",
                            "right": "0",
                            "bottom": "0",
                            "color": "rgba(255, 255, 255, 0.8)",
                            "font-size": "18px",
                            "font-weight": "regular",
                            "padding": "5px 10px",
                            "font-style": "italic",
                            "border-radius": "5px",
                        },
                    ),
                ]
            else:
                metrics_children = [html.Span("Disabled", style={'color': '#888'})]

            stats = get_system_stats()
            if stats is None:
                system_children = [html.Span("CPU stats need Linux", style={'color': '#888', 'padding': '5px 10px'})]
            else:
                stat_style = {
                    "position": "absolute",
                    "bottom": "0",
                    "color": "rgba(255, 255, 255, 0.8)",
                    "font-size": "16px",
                    "font-style": "italic",
                    "padding": "5px 10px",
                }
                per_core = " · ".join(f"cpu{i} {p:.0f}%" for i, p in enumerate(stats["cores"]))
                temp = stats["temp_c"]
                system_children = [
                    html.Span(
                        f"CPU: {stats['cpu']:.0f}% (busiest core {max(stats['cores'], default=0):.0f}%)",
                        title=per_core,
                        style={**stat_style, "left": "0"},
                    ),
                    html.Span(
                        f"CPU Temp: {temp:.1f} °C" if temp is not None else "CPU Temp: n/a",
                        style={**stat_style, "right": "0"},
                    ),
                ]

            if not pipeline or not pipeline.is_enabled():
                detection_items = [html.Div("Camera disabled", style={'color': '#888'})]
            else:
                detections = pipeline.get_detections()
                if not detections:
                    detection_items = [html.Div("No detections", style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'text-align': 'center',
                        'border-radius': '10px',
                        'border': '2px solid rgba(255, 255, 255, 0.5)',
                        'padding': '10px',
                        'margin': '0 0px 20px 20px',
                    })]
                else:
                    detection_items = []
                    for i, det in enumerate(detections):
                        if pipeline.get_pipeline_type() == "apriltag":
                            detection_items.append(html.Div([
                                html.Span(f"Detection #{i + 1}:", style={'color': '#CCC9CA', 'margin-right': '10px', 'font-weight': 'medium'}),
                                html.Span(f"Tag {det.getTag()}", style={'color': '#CCC9CA', 'margin-right': '5px'}),
                                html.Span(f"R {det.getR():.1f}in", style={'color': '#CCC9CA', 'margin-right': '5px'}),
                                html.Span(f"θ {det.getTheta():.2f}deg", style={'color': '#CCC9CA'}),
                            ], style={
                                'border': '2px solid rgba(255, 255, 255, 0.5)',
                                'border-radius': '10px',
                                'padding': '10px',
                                'font-size': '14px',
                                'margin': '0 0px 20px 20px',
                            }))
                        else:
                            detection_items.append(html.Div([
                                html.Span(f"Detection #{i + 1}:", style={'color': '#CCC9CA', 'margin-right': '10px', 'font-weight': 'medium'}),
                                html.Span(f"R {det.getR():.2f}in", style={'color': '#CCC9CA', 'margin-right': '5px'}),
                                html.Span(f"θ {det.getTheta():.2f}deg", style={'color': '#CCC9CA', 'margin-right': '10px'}),
                                html.Span(f"On: {det.isOnTop()}", style={'color': '#CCC9CA'}),
                            ], style={
                                'border': '2px solid rgba(255, 255, 255, 0.5)',
                                'border-radius': '10px',
                                'padding': '10px',
                                'font-size': '14px',
                                'margin': '0 0px 20px 20px',
                            }))

            return metrics_children, system_children, detection_items
    
        self.setup_slider_callbacks()
        
        self.server.add_url_rule(f'/video_feed/{cam_id}', f'video_feed_{cam_id}',
                                lambda cid=cam_id: self.video_feed(cid))
        self.server.add_url_rule(f'/calibration_feed/{cam_id}', f'calibration_feed_{cam_id}',
                                lambda cid=cam_id: self.calibration_feed(cid))

        @self.server.route('/api/calibration/health')
        def api_calibration_health():
            pipeline = self.vision_manager.get_pipeline(cam_id)
            cal_status = self.calibrator.get_status()
            health = (
                pipeline.get_health_report()
                if pipeline is not None and hasattr(pipeline, "get_health_report")
                else cal_status.get("health", {})
            )
            return jsonify({
                "camera_id": cam_id,
                "health": health,
                "calibration_status": cal_status,
            })

    def setup_slider_callbacks(self):
        from dash import no_update
        from localization.detection import SET_THRESH_STEP, SET_THRESH_WIN
        from localization.visiony import CONF, ASPECT_THRESH
        from localization.gamepiece_solution import WD, ONT

        for slider_type, setter in (
            ("dfc-target", _preprocess_setter("dfc-target")),
            ("aslc-bins", _preprocess_setter("aslc-bins")),
            ("aslc-min-corr", _preprocess_setter("aslc-min-corr")),
            ("divergence-gain", _preprocess_setter("divergence-gain")),
            ("thresh-step", SET_THRESH_STEP),
            ("max-thresh", SET_THRESH_WIN),
            ("framecomp", DashboardServer.framecomp_slider.setFloat),
            ("yolo_conf", CONF.setFloat),
            ("yolo_aspect", ASPECT_THRESH.setFloat),
            ("wd", WD.setFloat),
            ("ont", ONT.setFloat),
        ):
            @self.app.callback(
                Output({"type": slider_type, "index": MATCH}, "value"),
                [Input({"type": slider_type, "index": MATCH}, "value")],
                prevent_initial_call=True,
            )
            def _update_slider(value, _setter=setter):
                if value is not None:
                    _setter(value)
                return value

        @self.app.callback(
            [Output({'type': 'gain', 'index': MATCH}, 'disabled'),
             Output({'type': 'retune-gain', 'index': MATCH}, 'disabled'),
             Output({'type': 'exposure-status', 'index': MATCH}, 'children')],
            [Input({'type': 'auto-exposure', 'index': MATCH}, 'value'),
             Input({'type': 'exposure', 'index': MATCH}, 'value'),
             Input({'type': 'gain', 'index': MATCH}, 'value'),
             Input({'type': 'retune-gain', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def update_exposure(auto_value, exposure, gain, _retune_clicks):
            auto = 'auto' in (auto_value or [])
            pipeline = self.vision_manager.get_pipeline(self.camera_id)
            if not pipeline or exposure is None or gain is None:
                return auto, not auto, no_update
            if (ctx.triggered_id or {}).get('type') == 'retune-gain':
                pipeline.cam.retune_gain()
            else:
                pipeline.cam.set_exposure_settings(auto, int(exposure), int(gain))
            pipeline.cam.wait_for_gain()
            return auto, not auto, pipeline.cam.get_exposure_settings()["status"]

        @self.app.callback(
            Output({'type': 'resolution-status', 'index': MATCH}, 'children'),
            [Input({'type': 'resolution', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_resolution(value):
            pipeline = self.vision_manager.get_pipeline(self.camera_id)
            if not pipeline or not value:
                return no_update
            width, height = map(int, value.split("x"))
            if (width, height) != pipeline.cam.get_resolution():
                pipeline.cam.set_resolution(width, height)
                if self._calibrator is not None:
                    self._calibrator.reset()
            return self._resolution_status(self.camera_id, width, height)

        @self.app.callback(
            Output({'type': 'reboot', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'reboot', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def reboot_system(n_clicks):
            if n_clicks:
                os.system('sudo reboot')
            return n_clicks

        @self.app.callback(
            Output({'type': 'reload', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'reload', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def reload_pipeline(n_clicks):
            if n_clicks:
                Thread(target=self.vision_manager.reload_all_pipelines, daemon=True).start()
            return n_clicks

        self.setup_calibration_callbacks()
        self.setup_focus_callbacks()

    def setup_focus_callbacks(self):
        cam_id = self.camera_id
        
        @self.app.callback(
            Output(f'focus-score-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_focus_score(n_intervals):
            pipeline = self.vision_manager.get_pipeline(cam_id)
            if pipeline and pipeline.focus_mode:
                return f"Score: {pipeline.focus_score:.1f}"
            return "Focus Mode Disabled"
            
        @self.app.callback(
            Output({'type': 'toggle-focus', 'index': MATCH}, 'children'),
            [Input({'type': 'toggle-focus', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def toggle_focus_mode(n_clicks):
            if n_clicks:
                pipeline = self.vision_manager.get_pipeline(cam_id)
                if pipeline:
                    pipeline.focus_mode = not pipeline.focus_mode
                    return "Disable Focus Mode" if pipeline.focus_mode else "Enable Focus Mode"
            from dash import no_update
            return no_update

    def setup_calibration_callbacks(self):
        cam_id = self.camera_id
        
        @self.app.callback(
            Output(f'calibration-status-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_calibration_status(n_intervals):
            status = self.calibrator.get_status()
            progress_pct = status['progress'] * 100
            progress_color = '#00ff00' if status['is_calibrating'] else '#ffaa00'
            health = status.get('health') or {}
            h_status = health.get('status', 'UNKNOWN')
            h_color = '#00ff00' if h_status == 'OK' else ('#ffaa00' if h_status in ('WARNING', 'UNKNOWN', 'NO_BOARD') else '#ff5555')
            h_metrics = health.get('metrics') or {}
            h_issues = health.get('issues') or []

            health_nodes = [
                html.Div(
                    f"Camera Health Self-Check: {h_status} - {health.get('summary', '')}",
                    style={'margin': '8px 0 4px 0', 'color': h_color, 'font-weight': 'bold'},
                )
            ]
            if h_metrics:
                health_nodes.append(html.Div(
                    f"Sharpness: {h_metrics.get('edge_sharpness', 0):.2f} | "
                    f"Blur (Lap): {h_metrics.get('blur_score', 0):.0f} | "
                    f"Contrast: {h_metrics.get('contrast', 0):.0f} | "
                    f"Noise: {h_metrics.get('noise_sigma', 0):.1f}",
                    style={'margin': '2px 0', 'font-size': '12px', 'color': '#CCC9CA'},
                ))
            for issue in h_issues[:2]:
                health_nodes.append(html.Div(
                    f"[{issue.get('severity')}] {issue.get('message')} -> {issue.get('recommendation')}",
                    style={'margin': '2px 0', 'font-size': '12px', 'color': '#ff8888' if issue.get('severity') == 'ERROR' else '#ffd166'},
                ))

            # Show progress bar during calibration
            progress_bar = None
            if status['is_calibrating']:
                progress_bar = html.Div([
                    html.Div(style={
                        'width': f'{progress_pct:.1f}%',
                        'height': '20px',
                        'background-color': '#00ff00',
                        'transition': 'width 0.3s ease',
                        'border-radius': '4px',
                    }),
                ], style={
                    'width': '100%',
                    'height': '20px',
                    'background-color': '#333',
                    'border-radius': '4px',
                    'overflow': 'hidden',
                    'margin': '5px 0',
                })
            
            return html.Div([
                html.Div(f"Status: {status['status_message']}", style={'margin': '5px 0'}),
                html.Div(f"Samples: {status['num_samples']}/{status['target_samples']}", style={'margin': '5px 0'}),
                html.Div(f"Images: {status['sample_dir']}", style={'margin': '5px 0'}) if status['sample_dir'] else html.Div(),
                html.Div(f"Progress: {progress_pct:.1f}%", style={'margin': '5px 0'}),
                progress_bar if progress_bar else html.Div(),
                html.Div(
                    "✓ Ready to calibrate" if status['is_ready'] and not status['is_calibrating'] else 
                    ("Calibrating..." if status['is_calibrating'] else "Need more samples"),
                    style={'margin': '5px 0', 'color': progress_color}
                ),
                *health_nodes,
            ])
        
        @self.app.callback(
            Output(f'calibration-results-{cam_id}', 'children'),
            Output(f'calibration-results-{cam_id}', 'style'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_calibration_results(n_intervals):
            result = self.calibrator.calibration_result
            if result is None:
                return [], {'display': 'none'}
            
            meta = result.get('meta', {})
            fov = meta.get('fov', {})
            distortion_viz = meta.get('distortion_viz_base64', '')
            diag = meta.get('diagnostics') or {}
            diag_status = diag.get('status', 'OK')
            diag_color = '#00ff00' if diag_status == 'OK' else ('#ffaa00' if diag_status == 'WARNING' else '#ff5555')
            diag_issues = [
                html.Div(
                    f"[{i.get('severity')}] {i.get('message')} -> {i.get('recommendation')}",
                    style={'color': '#ff8888' if i.get('severity') == 'ERROR' else '#ffd166', 'margin': '4px 0', 'font-size': '13px'},
                )
                for i in (diag.get('issues') or [])
            ]
            
            return [
                html.H4("Calibration Results", style={
                    'color': '#CDA646',
                    'margin': '20px 0 10px 0',
                    'text-align': 'center',
                }),
                html.Div([
                    html.Div([
                        html.H5("Field of View & Optical Health", style={'color': '#CCC9CA', 'margin': '10px 0'}),
                        html.Div(f"Horizontal: {fov.get('horizontal', 0):.1f}°", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(f"Vertical: {fov.get('vertical', 0):.1f}°", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(f"Diagonal: {fov.get('diagonal', 0):.1f}°", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(f"Reprojection Error: {meta.get('reprojection_error', 0):.4f} px", 
                                style={'color': '#CCC9CA', 'margin': '15px 0 5px 0'}),
                        html.Div(f"Samples Used: {meta.get('num_samples', 0)}", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(
                            f"Optical Self-Check: {diag_status} - {diag.get('summary', '')}",
                            style={'color': diag_color, 'margin': '10px 0 4px 0', 'font-weight': 'bold'},
                        ),
                        *diag_issues,
                    ], style={
                        'flex': '1',
                        'padding': '20px',
                        'border': '2px solid #CDA646',
                        'border-radius': '10px',
                        'margin': '10px',
                    }),
                    html.Div([
                        html.Img(src=f"data:image/png;base64,{distortion_viz}",
                                style={
                                    'max-width': '100%',
                                    'border': '2px solid #CDA646',
                                    'border-radius': '5px',
                                }),
                    ], style={
                        'flex': '2',
                        'padding': '20px',
                        'margin': '10px',
                    }),
                ], style={
                    'display': 'flex',
                    'flex-direction': 'row',
                    'align-items': 'center',
                    'justify-content': 'center',
                }),
                html.Div([
                    html.Button("Save Calibration", id={'type': 'save-calibration', 'index': cam_id}, style={
                        "margin": "20px 5px",
                        "font-size": "16px",
                        "color": "#161616",
                        "background-color": "rgba(100, 200, 100, 1)",
                        "border": "none",
                        "padding": "12px 24px",
                        "width": "250px",
                        "height": "50px",
                        "border-radius": "25px",
                        "cursor": "pointer",
                        "font-weight": "bold"
                    }),
                ], style={
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                })
            ], {
                'width': '100%',
                'padding': '20px',
                'display': 'block',
                'background-color': 'rgba(0, 0, 0, 0.3)',
                'border-top': '2px solid #CDA646',
            }
        
        @self.app.callback(
            Output({'type': 'run-calibration', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'run-calibration', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def run_calibration(n_clicks):
            if n_clicks:
                Thread(target=self.calibrator.calibrate, daemon=True).start()
            return n_clicks
        
        @self.app.callback(
            Output({'type': 'reset-calibration', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'reset-calibration', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def reset_calibration(n_clicks):
            if n_clicks:
                self.calibrator.reset()
            return n_clicks
        
        @self.app.callback(
            Output({'type': 'save-cal-samples', 'index': MATCH}, 'value'),
            [Input({'type': 'save-cal-samples', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_save_samples(value):
            save_samples.setInt(1 if 'save' in (value or []) else 0)
            return value

        @self.app.callback(
            Output({'type': 'save-calibration', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'save-calibration', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def save_calibration(n_clicks):
            if n_clicks:
                self.calibrator.save_calibration()
            return n_clicks
        
    def video_feed(self, camera_id: int):
        return Response(self.generate_frames(camera_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames(self, camera_id: int):
        return _stream_mjpeg_frames(
            self.vision_manager,
            camera_id,
            lambda: int(DashboardServer.framecomp_slider.valueFloat() * 100),
            "Camera Disabled",
            (200, 240),
        )

    def calibration_feed(self, camera_id: int):
        return Response(self.generate_calibration_frames(camera_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_calibration_frames(self, camera_id: int):
        blank_frame_cache = None
        last_frame = None
        yield MJPEG_PART_HEADER

        while True:
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                if blank_frame_cache is None:
                    blank_frame_cache = _build_blank_jpeg("Camera Not Available", (180, 240), quality=60)
                yield mjpeg_part(blank_frame_cache)
                time.sleep(0.5)
                continue

            frame = pipeline.cam.get_raw_frame()
            annotated_frame = None
            if frame is not None:
                try:
                    annotated_frame, _, _ = self.calibrator.process_frame(frame, auto_capture=True)
                except Exception as e:
                    import traceback
                    from util.logger import Logger
                    log = Logger("DashboardServer")
                    log.Error(f"Error processing calibration frame: {e}\n{traceback.format_exc()}")

            if annotated_frame is None:
                if last_frame is not None:
                    yield mjpeg_part(last_frame)
                time.sleep(0.05)
                continue

            if self.calibrator.try_auto_capture():
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 255, 0), 10)
                cv2.putText(annotated_frame, "CAPTURED!", (annotated_frame.shape[1] // 2 - 100, annotated_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            if ret:
                last_frame = buffer.tobytes()
                yield mjpeg_part(last_frame)
            time.sleep(0.03)  # ~30 FPS

    def start_server(self):
        pin_current_thread_little_cores()
        port = dashboard_port(self.camera_id)
        print(f"\n{'='*50}")
        print(f"MonkeySee Dashboard Starting for Camera {self.camera_id}")
        print(f"Port: {port}")
        print(f"Access at: http://0.0.0.0:{port}")
        print(f"{'='*50}\n")
        self.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    def start_server_thread(self):
        Thread(target=self.start_server, daemon=True).start()

    def index_string(self):
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MonkeyVision</title>
            <style>
                body {
                    background-color: #161616;
                    color: white;
                    font-family: 'Inter', sans-serif;
                    font-size: 15px;
                    margin: 0;
                    padding: 0;
                }
                .funky-slider .rc-slider-track {
                    background-color: #CDA646;
                }
                .funky-slider .rc-slider-rail {
                    background-color: rgba(255, 255, 255, 0.5);
                }
                .funky-slider .rc-slider-handle {
                    border-color: #CDA646;
                    background-color: #CDA646;
                }
                .funky-slider .rc-slider-tooltip {
                    font-size: 14px;
                    color: rgba(255, 255, 255, 0.5);
                    background-color: #CDA646;
                    border-radius: 8px;
                    box-shadow: none;
                }
                .custom-tab {
                    background-color: #161616 !important;
                    color: #CCC9CA !important;
                    border: none !important;
                    padding: 12px 24px !important;
                    font-size: 16px !important;
                    font-weight: 500 !important;
                    border-bottom: 3px solid transparent !important;
                }
                .custom-tab:hover {
                    background-color: #2a2a2a !important;
                }
                .custom-tab--selected {
                    background-color: #161616 !important;
                    color: #CDA646 !important;
                    border-bottom: 3px solid #CDA646 !important;
                }
                ::-webkit-scrollbar {
                    width: 10px;
                }
                ::-webkit-scrollbar-track {
                    background: transparent;
                }
                ::-webkit-scrollbar-thumb {
                    background: #CDA646;
                    border-radius: 5px;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #b8923d;
                }
                * {
                    scrollbar-width: thin;
                    scrollbar-color: #CDA646 transparent;
                }
            </style>
            <script>
                document.addEventListener('click', function(e) {
                    var el = e.target.closest('.camera-nav-link, .camera-hub-link');
                    if (!el) return;
                    var port = el.getAttribute('data-port');
                    if (!port) return;
                    e.preventDefault();
                    window.location.href = 'http://' + window.location.hostname + ':' + port + '/';
                });
            </script>
        </head>
        <body>
            {%app_entry%}
            {%config%}
            {%scripts%}
            {%renderer%}
        </body>
        </html>
        """


class CameraIndexServer:
    """Hub on port 5800 listing every camera stream with capture controls."""

    def __init__(self, vision_manager: VisionManager):
        self.vision_manager = vision_manager
        self.server = Flask("MonkeySeeCameraIndex")
        self.setup_routes()
        Thread(target=self.start_server, daemon=True).start()

    def start_server(self):
        pin_current_thread_little_cores()
        print(f"\n{'='*50}")
        print("MonkeySee Camera Hub Starting")
        print(f"Port: {DASHBOARD_BASE_PORT}")
        print(f"Access at: http://0.0.0.0:{DASHBOARD_BASE_PORT}")
        print(f"{'='*50}\n")
        self.server.run(host="0.0.0.0", port=DASHBOARD_BASE_PORT, debug=False, use_reloader=False)

    def setup_routes(self):
        @self.server.route("/")
        def index():
            return self._index_html()

        @self.server.route("/api/cameras")
        def api_cameras():
            cameras = []
            for cam_id, pipeline in sorted(self.vision_manager.get_all_pipelines().items()):
                status = pipeline.get_capture_status()
                enabled = pipeline.is_enabled()
                backend_fps = pipeline.get_backend_fps() if enabled else 0.0
                framerate = pipeline.get_framerate() if enabled else 0.0
                fps = pipeline.get_fps() if hasattr(pipeline, "get_fps") else (backend_fps if backend_fps > 0 else framerate)
                proc_latency_ms = pipeline.get_processing_latency() * 1000.0 if enabled else 0.0
                cap_latency_ms = (
                    pipeline.get_capture_latency() * 1000.0
                    if enabled and hasattr(pipeline, "get_capture_latency")
                    else 0.0
                )
                resolution = (
                    pipeline.get_resolution()
                    if hasattr(pipeline, "get_resolution")
                    else pipeline.cam.get_resolution()
                )
                det_summary = (
                    pipeline.get_detection_summary()
                    if enabled and hasattr(pipeline, "get_detection_summary")
                    else []
                )
                health = (
                    pipeline.get_health_report()
                    if hasattr(pipeline, "get_health_report")
                    else {"status": "OK", "ok": True, "summary": "", "issues": [], "metrics": {}}
                )
                cameras.append({
                    "id": cam_id,
                    "name": pipeline.get_camera_name(),
                    "pipeline": pipeline.get_pipeline_type(),
                    "enabled": enabled,
                    "dashboard_port": dashboard_port(cam_id),
                    "recording": status["recording"],
                    "message": status["message"],
                    "path": status["path"],
                    "fps": round(fps, 2),
                    "backend_fps": round(backend_fps, 2),
                    "framerate": round(framerate, 2),
                    "latency_ms": round(proc_latency_ms, 2),
                    "capture_latency_ms": round(cap_latency_ms, 2),
                    "resolution": [int(resolution[0]), int(resolution[1])],
                    "detection_count": len(det_summary),
                    "detections": det_summary,
                    "health": health,
                })
            return jsonify({"cameras": cameras})

        @self.server.route("/video_feed/<int:camera_id>")
        def video_feed(camera_id: int):
            return Response(
                self._generate_frames(camera_id),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.server.route("/api/<int:camera_id>/snapshot", methods=["POST"])
        def api_snapshot(camera_id: int):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                return jsonify({"ok": False, "message": "Camera not found"}), 404
            path = pipeline.save_snapshot()
            status = pipeline.get_capture_status()
            return jsonify({"ok": path is not None, "path": path, **status})

        @self.server.route("/api/<int:camera_id>/record/start", methods=["POST"])
        def api_record_start(camera_id: int):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                return jsonify({"ok": False, "message": "Camera not found"}), 404
            message = pipeline.start_recording()
            status = pipeline.get_capture_status()
            return jsonify({"ok": True, "message": message, **status})

        @self.server.route("/api/<int:camera_id>/record/stop", methods=["POST"])
        def api_record_stop(camera_id: int):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                return jsonify({"ok": False, "message": "Camera not found"}), 404
            message = pipeline.stop_recording()
            status = pipeline.get_capture_status()
            return jsonify({"ok": True, "message": message, **status})

    def _generate_frames(self, camera_id: int):
        return _stream_mjpeg_frames(
            self.vision_manager,
            camera_id,
            lambda: 70,
            "Camera Unavailable",
            (140, 240),
        )

    def _index_html(self):
        cards = []
        for cam_id, pipeline in sorted(self.vision_manager.get_all_pipelines().items()):
            enabled = pipeline.is_enabled()
            status = "Live" if enabled else "Disabled"
            cards.append(
                f"""
                <article class="card" data-cam="{cam_id}">
                  <div class="card-head">
                    <div>
                      <h2>{pipeline.get_camera_name()}</h2>
                      <p class="meta" id="meta-{cam_id}">Camera {cam_id} · {pipeline.get_pipeline_type()} · {status}</p>
                    </div>
                    <a class="dash-link" href="http://HOST:{dashboard_port(cam_id)}/">Open dashboard</a>
                  </div>
                  <img class="stream" src="/video_feed/{cam_id}" alt="Camera {cam_id} stream" />
                  <div class="actions">
                    <button type="button" data-action="snapshot" data-cam="{cam_id}">Snapshot</button>
                    <button type="button" data-action="record-start" data-cam="{cam_id}">Start Recording</button>
                    <button type="button" data-action="record-stop" data-cam="{cam_id}">Stop Recording</button>
                  </div>
                  <p class="status" id="status-{cam_id}">Ready</p>
                  <p class="health-line" id="health-{cam_id}">Health: Checking...</p>
                </article>
                """
            )

        cards_html = "\n".join(cards) if cards else "<p class='empty'>No cameras found in config.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MonkeyVision Cameras</title>
  <style>
    :root {{
      --bg: #161616;
      --text: #CCC9CA;
      --accent: #CDA646;
      --panel: #1f1f1f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 32px;
      color: var(--text);
    }}
    .subtitle {{
      margin: 0 0 24px;
      color: var(--text);
      opacity: 0.85;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}
    .card h2 {{
      margin: 0;
      font-size: 18px;
      color: var(--accent);
    }}
    .meta {{
      margin: 4px 0 0;
      font-size: 13px;
      opacity: 0.8;
    }}
    .dash-link {{
      color: #161616;
      background: var(--accent);
      text-decoration: none;
      font-weight: 700;
      font-size: 12px;
      padding: 8px 12px;
      border-radius: 16px;
      white-space: nowrap;
    }}
    .stream {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #000;
      border: 2px solid var(--accent);
      border-radius: 9px;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    button {{
      border: none;
      border-radius: 18px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      color: #161616;
      background: rgba(100, 200, 255, 1);
    }}
    button[data-action="record-start"] {{ background: rgba(100, 200, 100, 1); }}
    button[data-action="record-stop"] {{ background: rgba(255, 100, 100, 1); }}
    .status {{
      margin: 0;
      min-height: 1.2em;
      font-size: 13px;
      opacity: 0.9;
    }}
    .health-line {{
      margin: 0;
      min-height: 1.2em;
      font-size: 12px;
      font-weight: 600;
      opacity: 0.9;
    }}
    .health-ok {{ color: #7ae582; }}
    .health-warn {{ color: #ffd166; }}
    .health-err {{ color: #ff6b6b; }}
    .empty {{ opacity: 0.7; }}
  </style>
</head>
<body>
  <h1>MonkeyVision</h1>
  <p class="subtitle">Pick a camera stream to preview, snapshot, or record. Snapshots/recordings save under captures/.</p>
  <div class="grid">
    {cards_html}
  </div>
  <script>
    (function() {{
      const host = window.location.hostname;
      document.querySelectorAll('.dash-link').forEach(function(el) {{
        el.href = el.getAttribute('href').replace('HOST', host);
      }});

      document.addEventListener('visibilitychange', function() {{
        document.querySelectorAll('img.stream').forEach(function(img) {{
          if (document.hidden) {{
            img.dataset.feed = img.getAttribute('src');
            img.src = 'data:,';
          }} else if (img.dataset.feed) {{
            img.src = img.dataset.feed;
          }}
        }});
      }});

      async function refreshStatuses() {{
        try {{
          const res = await fetch('/api/cameras');
          const data = await res.json();
          (data.cameras || []).forEach(function(cam) {{
            const metaEl = document.getElementById('meta-' + cam.id);
            if (metaEl) {{
              if (cam.enabled) {{
                const res = (cam.resolution && cam.resolution.length === 2) ? (cam.resolution[0] + 'x' + cam.resolution[1]) : '';
                const fps = (typeof cam.fps === 'number') ? cam.fps.toFixed(1) + ' FPS' : '';
                const lat = (typeof cam.latency_ms === 'number') ? cam.latency_ms.toFixed(1) + ' ms' : '';
                const dets = (typeof cam.detection_count === 'number') ? cam.detection_count + ' det' : '';
                metaEl.textContent = ['Camera ' + cam.id, cam.pipeline, 'Live', res, fps, lat, dets].filter(Boolean).join(' · ');
              }} else {{
                metaEl.textContent = 'Camera ' + cam.id + ' · ' + cam.pipeline + ' · Disabled';
              }}
            }}
            const el = document.getElementById('status-' + cam.id);
            if (el) {{
              const state = cam.recording ? 'Recording' : 'Idle';
              const msg = cam.message || 'Ready';
              el.textContent = state + ' · ' + msg;
            }}
            const hEl = document.getElementById('health-' + cam.id);
            if (hEl && cam.health) {{
              const hs = cam.health.status || 'UNKNOWN';
              const hsum = cam.health.summary || '';
              hEl.textContent = 'Health: ' + hs + (hsum ? ' · ' + hsum : '');
              hEl.className = 'health-line ' + (hs === 'ERROR' ? 'health-err' : (hs === 'WARNING' ? 'health-warn' : 'health-ok'));
            }}
          }});
        }} catch (e) {{}}
      }}

      document.querySelectorAll('button[data-action]').forEach(function(btn) {{
        btn.addEventListener('click', async function() {{
          const cam = btn.getAttribute('data-cam');
          const action = btn.getAttribute('data-action');
          let url = '/api/' + cam + '/snapshot';
          if (action === 'record-start') url = '/api/' + cam + '/record/start';
          if (action === 'record-stop') url = '/api/' + cam + '/record/stop';
          try {{
            const res = await fetch(url, {{ method: 'POST' }});
            const data = await res.json();
            const el = document.getElementById('status-' + cam);
            if (el) el.textContent = data.message || (data.ok ? 'OK' : 'Failed');
          }} catch (e) {{
            const el = document.getElementById('status-' + cam);
            if (el) el.textContent = 'Request failed';
          }}
        }});
      }});

      refreshStatuses();
      setInterval(refreshStatuses, 1000);
    }})();
  </script>
</body>
</html>
"""

