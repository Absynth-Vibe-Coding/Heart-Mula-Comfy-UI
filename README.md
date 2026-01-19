<div align="center">
  <h1>HeartMuLa ComfyUI Nodes</h1>
<strong>Heart Music Language Model</strong>
  <p><em>Generate music from lyrics using HeartMuLa in ComfyUI</em></p>
</div>

<img width="1511" height="615" alt="image" src="https://github.com/user-attachments/assets/87f10bd3-46a5-49f3-87af-e9461659ce26" />


<h2>NEWS</strong></h2>
<strong>v1.1: added quantization options for lower vram</strong><br>
-int8 - ~50% VRAM reduction<br>
-int4 - ~75% VRAM reduction (uses nf4 quantization)<br><br>

Info: you can still get ooms, then just retry but its still going to be faster when it works using the quantization<br><br>
Try more codec steps now since its faster

--------------------------------------------------------------------------------------------------------------------------------

Download the [Model](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B)  
Download the : [Workflow](https://github.com/Absynth-Vibe-Coding/Heart-Mula-Comfy-UI/blob/main/HeartMula%20Absynth%20v1.0.json)
<br>If you like this: Subscribe to my YouTube Channel https://www.youtube.com/@Electric-Dreamz

<strong>Important:</strong> dont sell the music 
🔒 For non-commercial research and educational use only<br>
🚫 Any commercial use is strictly prohibited


<h2>About</h2>

<p>Custom nodes for <a href="https://heartmula.github.io/">HeartMuLa</a> - an open-source AI music generation model. Write lyrics, pick a style, and generate full songs directly in ComfyUI.</p>

<h2>Features</h2>

- select genre preset (uses tags)
- custom tags: seperate with comma
- adjust codec steps: 1 is already very good, but you can go higher (better quality)

<img width="756" height="823" alt="image" src="https://github.com/user-attachments/assets/bef44d3e-00d4-42f8-a1c8-5237c8dfe712" />


<ul>
  <li><strong>Genre Presets</strong> - EDM, Hip Hop, Rock, Jazz, Lo-Fi, and more</li>
  <li><strong>Custom Tags</strong> - Full control over style with comma-separated tags</li>
  <li><strong>Adjustable Duration</strong> - Generate 5 seconds to 4 minutes of audio</li>
  <li><strong>Windows Compatible</strong> - Includes torchcodec workaround</li>
</ul>

<h2>Installation</h2>

<ol>
  <li>
    <p>Clone into <code>ComfyUI/custom_nodes/</code>:</p>
    <pre>git clone https://github.com/Absynth-Vibe-Coding/Heart-Mula-Comfy-UI.git</pre>
  </li>
  <li>
    <p>Install dependencies:</p>
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li>
    <pre>pip install --no-deps git+https://github.com/HeartMuLa/heartlib.git</pre></p>
  </li>
  <li>
    <p>Download the model from https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B and place in:</p>
    <pre>ComfyUI/models/heartmula/</pre>
  </li>
  <li>
    <p>Restart ComfyUI</p>
  </li>
</ol>

 The folder structure should be:
   <pre><code>models/heartmula/
  └── [model_name]/
      ├── HeartMuLa-oss-3B/
      │   └── config.json (+ other files)
      ├── HeartMuLa-oss-7B/
      │   └── config.json (+ other files)
      ├── HeartCodec-oss/
      ├── tokenizer.json
      └── gen_config.json
  </code></pre>
<hr>

<h2>Nodes</h2>

<h3>HeartMuLa Loader</h3>
<table>
  <tr><th>Parameter</th><th>Description</th></tr>
  <tr><td><code>model_name</code></td><td>Select downloaded model from models/heartmula</td></tr> Yes, you need all the files
  <tr><td><code>version</code></td><td>3B (faster, ~12GB VRAM) or 7B (better quality, more VRAM)</td></tr>
</table>

<h3>HeartMuLa Generate</h3>
<table>
  <tr><th>Parameter</th><th>Description</th></tr>
  <tr><td><code>lyrics</code></td><td>Song lyrics with markers like <code>[verse]</code>, <code>[chorus]</code></td></tr>
  <tr><td><code>genre</code></td><td>Select preset or "custom" to use custom_tags</td></tr>
  <tr><td><code>custom_tags</code></td><td>Comma-separated tags (no spaces after commas)</td></tr>
  <tr><td><code>max_duration_sec</code></td><td>Maximum audio length in seconds</td></tr>
  <tr><td><code>temperature</code></td><td>Creativity (higher = more varied)</td></tr>
  <tr><td><code>topk</code></td><td>Sampling parameter (default 50)</td></tr>
  <tr><td><code>cfg_scale</code></td><td>Guidance scale (default 1.0)</td></tr>
  <tr><td><code>codec_steps</code></td><td>Audio quality (1 = fast, 5+ = better)</td></tr>
  <tr><td><code>seed</code></td><td>0 = random, or set for reproducibility</td></tr>
</table>

<h2>Tags Format</h2>

<pre>trance,electronic,synthesizer,energetic,driving</pre>

<h2>Requirements</h2>

<ul>
  <li>ComfyUI</li>
  <li>~14GB disk space for model</li>
  <li>~12GB+ VRAM (3B) or ~24GB+ (7B)</li>
  <li>Python 3.10+</li>
</ul>

<h2>Troubleshooting</h2>

<table>
  <tr><th>Issue</th><th>Solution</th></tr>
  <tr><td>"TorchCodec is required"</td><td>Already patched in this node</td></tr>
  <tr><td>"No models found"</td><td>Check model folder structure: <code>models/heartmula/[name]/HeartMuLa-oss-3B/</code></td></tr>
  <tr><td>Wrong genre output</td><td>Verify tags format - no spaces after commas</td></tr>
  <tr><td>Slow generation</td><td>Set <code>codec_steps</code> to 1</td></tr>
</table>

<h2>Credits</h2>

<ul>
  <li><a href="https://github.com/HeartMuLa/heartlib">HeartMuLa</a> - Original model and library</li>
  <li><a href="https://heartmula.github.io/">HeartMuLa Project Page</a></li>
</ul>

<h2>License</h2>

<p>MIT</p>

