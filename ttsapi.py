from flask import Flask, request, send_file
import os
import subprocess
import locale
import tempfile
import math
import torch
import numpy as np
from scipy.io import wavfile
from data_utils import preprocess_text, TextMapper
from models import SynthesizerTrn

app = Flask(__name__)

# Define global variables for the model and text mapper
vocab_file = None
config_file = None
text_mapper = None
net_g = None
hps = None

# Function to initialize model and text mapper
def initialize_model():
    global vocab_file, config_file, text_mapper, net_g, hps

    # Download language model if not already downloaded
    if not os.path.exists("vits"):
        download()

    # Initialize necessary variables
    global LANG
    LANG = request.args.get('lang', default='eng')
    ckpt_dir = download(LANG)
    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net_g.to(device)
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"
    print(f"load {g_pth}")
    _ = utils.load_checkpoint(g_pth, net_g, None)

# Route for generating audio from text
@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    global vocab_file, config_file, text_mapper, net_g, hps

    Text = "I am sorry, but this Ramdhan I got so much distracted.h" #@param {type:"string"}
    # Get text input from request
    text = request.form.get('text')

    # Preprocess text
    txt = preprocess_text(text, text_mapper, hps, lang=LANG)
    stn_tst = text_mapper.get_text(txt, hps)

    # Generate audio from text
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0,0].cpu().float().numpy()

    # Save audio as WAV file
    output_path = "generated_audio.wav"
    wavfile.write(output_path, hps.data.sampling_rate, hyp.astype(np.int16))

    # Return WAV file
    return send_file(output_path, mimetype="audio/wav")

if __name__ == "__main__":
    initialize_model()
    app.run(host='0.0.0.0', port=80)
