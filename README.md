# GPT-Script
Small and simple Python script for playing with GPT and T0 models.

## Requirements
GPT-Script has been tested with Python 3.10 and PyTorch 1.12.

## Usage
Grab requirements by running command `pip install -r requirements.txt` or `pip install --user -r requirements.txt`.

Run the script with Python by running command `python gptscript.py` while in same directory (assuming you only have single version of Python installed).

## English commands and default values
* `.read` - Reads a file (preferably text) as an input. Example: `.read sample.txt`
* `.addread` - Reads a file and asks for additional input. Example: `.addread sample.txt`
* `.multiline` - Multiline input, press Ctrl + Z and Enter when done.
* `.stop`, `.exit`, `.quit` - Exit from GPT-Script.
* `.reset` - Clear the current history for model.
* `.count` - (1) Amount of outputs generated for same input. Example: `.count 3`
* `.length` - (100) Maximum length for output, doesn't actually seem to work. Example: `.length 150`
* `.temp` - (0.9) Temperature parameter, meaning how "creative" the output can be. Lower value means the model will have less creative outputs. Example: `.temp 0.7`
* `.lenp` - (1.0) Length penalty, higher value penalizes outputs with longer length, favoring shorter ones. Example: `.lenp 1.3`
* `.repp` - (1.0) Repetition penalty, higher value penalizes repetition in output, favoring different words. Example: `.repp 1.3`
* `.topk` - (50) Top-K sampling, in super simple words; how many words should be considered for output. Example: `.topk 80`
* `.topp` - (0.9) Top-P sampling, in super simple words as well; filters out words with probability, threshold being Top-P value. Example: `.topp 0.95`
* `.sampling` - (True) Toggles whether to use Greedy or Top-P + Top-K sampling (True is Top-P + Top-K).
* `.cutsentence` - (True) Toggles whether to attempt to cut sentence after each output (True is cutting).
* `.help`, `.?` - Prints help text.
