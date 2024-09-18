import gradio as gr
import cv2

def turn_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

iface = gr.Interface()