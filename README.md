<div align="center">
  <h1>HeartMuLa ComfyUI Nodes</h1>
<strong>Heart Music Language Model</strong>
  <p><em>Generate music from lyrics using HeartMuLa in ComfyUI</em></p>
</div>

<img width="1511" height="615" alt="image" src="https://github.com/user-attachments/assets/87f10bd3-46a5-49f3-87af-e9461659ce26" />


<h2>NEWS</h2>

Added Flash Attn. Support, on my 5060TI with Flash Attn. 2.7.4 i have a speedboost of about 30 sec. now.<br>
Huge benefit! now you can switch to int8 quality while still haveing fast gneration speed using the flash attention even on my 5060ti! <br>

It detects if you have flash attn installed, else its using sdpa


----------------------------------------------------------------------------------------------------------------------------------

Fixed several bugs. If you had issues with the node try it again now :-)<br>
Thx to user Illymir for testing!

Resolved an issue where switching cfg_scale would cause crashes with "batch size mismatch" or "mat1 and mat2 shapes cannot be multiplied" errors. The model's internal state was getting corrupted when reusing cached models with different CFG settings. <br>
Model caching is now disabled to ensure clean state for each generation. <br>

---------------------------------------------------------------------------------------------------------------------------------


v1.3 Added Rap to the preset list, removed not supported genre presets and made some finetuning, could reproduce Rap and EDM now with the new workflow settings.
For Rap Temp 1 could work better

---------------------------------------------------------------------------------------------------------------------------------

<strong>v1.2:</strong> fixed the ooms so far: model is moved to CPU before the codec decoding runs. This frees up several GB of VRAM 😎
<br>improved the preset system and my workflow. cfg 1.5 - 2 or higher recommended to follow the presets better <- but then it will be slower.<br>
Even on 16 gb Vram you can now also do higher codec steps <-better quality

-------------------------------------------------------------------------------------------------------------------------------

<strong>v1.1:</strong> added quantization options for lower vram<br>
-int8 - ~50% VRAM reduction<br>
-int4 - ~75% VRAM reduction (uses nf4 quantization) <-fastest <br><br> 

Info: you can still get ooms, then just retry but its still going to be faster when it works using the quantization<br><br>
Try more codec steps now since its faster, depends on how much vram you have. on 16gb vram use codec steps 1 or 2 else you probably get ooms
If it starts geting really slow then restart comfy.

--------------------------------------------------------------------------------------------------------------------------------

Download the [Model](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B) You need all of the files<br>
Download the Workflow
<br>If you like this: Subscribe to my YouTube Channel https://www.youtube.com/@Electric-Dreamz

<strong>Important:</strong> dont sell the music 
🔒 For non-commercial research and educational use only<br>
🚫 Any commercial use is strictly prohibited


<h2>About</h2>

<p>Custom nodes for <a href="https://heartmula.github.io/">HeartMuLa</a> - an open-source AI music generation model. Write lyrics, pick a style, and generate full songs directly in ComfyUI.</p>

<h2>Features</h2>

- select genre preset (uses tags)
- custom tags: seperate with comma, example for rap: male voice, rap, hip hop, boombap
- adjust codec steps: 1 is already very good, but you can go higher (better quality)

<img width="733" height="277" alt="image" src="https://github.com/user-attachments/assets/db47176f-cc65-406a-aae8-385c63500574" />



<ul>
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

