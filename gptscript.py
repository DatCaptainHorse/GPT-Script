# MIT License
# 
# Copyright (c) 2021 Kristian Ollikainen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Language strings
languages = \
{ 
	"en": { 
			"id_language": "en",
			"menu_version": "GPT-Script version 1.1",
			"menu_info": "Options, '->' means currently selected and '--' means not selected",
			"menu_exit": "Exit",
			"menu_language": "Switch language: English",
			"menu_start": "Start with chosen parameters",
			"menu_usegpu": "Use CUDA (NVIDIA GPU) if available",
			"menu_models": "[Models]",
			"menu_input_option": "Option number: ",
			"error_unknowncommand": "Unknown command",
			"error_unknownfile": "Unknown file",
			"warn_cudanotavailable": "CUDA device not found, using CPU",
			"info_loading": "Loading..",
			"info_newlength": "Token length:",
			"info_output": "Output",
			"input_prompt": "Input: ",
			"input_additional": "Additional input: ",
			"input_multiline": "v Multiline input, press Ctrl + Z and Enter when done v",
		  },

	"fi": { 
			"id_language": "fi",
			"menu_version": "GPT-Script versio 1.1",
			"menu_info": "Vaihtoehdot, '->' tarkoittaa tällä hetkellä valittua ja '--' tarkoittaa valitsematonta",
			"menu_exit": "Poistu",
			"menu_language": "Vaihda kieltä: Suomi",
			"menu_start": "Aloita valituilla parametreillä",
			"menu_usegpu": "Käytä CUDA:a (NVIDIA näytönohjain) jos saatavilla",
			"menu_models": "[Mallit]",
			"menu_input_option": "Vaihtoehdon numero: ",
			"error_unknowncommand": "Tuntematon komento",
			"error_unknownfile": "Tuntematon tiedosto",
			"warn_cudanotavailable": "CUDA laitetta ei löytynyt, käytetään prosessoria",
			"info_loading": "Ladataan..",
			"info_newlength": "Token pituus:",
			"info_output": "Ulostulo",
			"input_prompt": "Syöte: ",
			"input_additional": "Lisäsyöte: ",
			"input_multiline": "v Monirivi syöte, paina Ctrl + Z ja Enter kun valmis v",
		  },
}

# Default language
language = languages["en"]

# Command strings (not restriced by chosen language so they work both ways)
commands = \
{
	"readfile": (".read", ".lue"),
	"readfile_additive": (".addread", ".lisälue"),
	"temperature": (".temp", ".lämpö"),
	"length_penalty": (".lenp", ".pituusr"),
	"repetition_penalty": (".repp", ".toistor"),
	"top_k_sampling": (".topk", ".huippuk"),
	"top_p_sampling": (".topp", ".huippup"),
	"wordlength": (".length", ".pituus"),
	"outputcount": (".count", ".määrä"),
	"resethistory": (".reset", ".tyhjää"),
	"exitscript": (".stop", ".exit", ".quit", ".poistu", ".sulje", ".lopeta"),
	"multilineinput": (".multiline", ".monirivi"),
}

# Defaults
useGPUifAvailable = False
GPTModels = { 
				"EleutherAI/gpt-neo-125M": [True, 0.5], 
				"EleutherAI/gpt-neo-1.3B": [False, 5], 
				"EleutherAI/gpt-neo-2.7B": [False, 10],
				"EleutherAI/gpt-j-6B": [False, 25],
				"bigscience/T0_3B": [False, 11]
			}

# Sets specific model as "selected" (True, others False)
# There is probably a better way to do this, however this works for now.
def selectModel(modelName):
	global GPTModels
	for k, v in GPTModels.items():
		v[0] = False
	GPTModels[modelName][0] = True

# Gets the first model with True value
def getSelectedModel():
	global GPTModels
	for k, v in GPTModels.items():
		if v[0]:
			return k

while True:
	print("\033[H\033[2J\033[H")
	print(language["menu_version"])
	print(language["menu_info"] + "\n")
	print("[0] " + language["menu_exit"])
	print("[1] " + language["menu_start"])
	print("[2] " + language["menu_language"])
	print(f"[3] { '->' if useGPUifAvailable else '--' } " + language["menu_usegpu"])
	print("\n" + language["menu_models"])

	modelOffset = 4
	idx = modelOffset
	for k, v in GPTModels.items():
		print(f"\t[{idx}] { '->' if v[0] else '--' } {k} (~{v[1]}GB RAM/VRAM)")
		idx += 1

	try:
		option = int(input("\n" + language["menu_input_option"]))
	except ValueError:
		continue

	if option == 0:
		exit()
	elif option == 1:
		break
	elif option == 2:
		iterator = iter(languages)
		for k in iterator:
			if k == language["id_language"]:
				language = languages[next(iterator, "en")]
	elif option == 3:
		useGPUifAvailable ^= 1
	else:
		for i, k in enumerate(GPTModels):
			if i == option - modelOffset:
				selectModel(k)
				break

# GPU availability check
if useGPUifAvailable and torch.cuda.is_available():
	usingGPU = True
	dataType = torch.float16
	dev = "cuda:0"
else:
	usingGPU = False
	dataType = torch.float32
	dev = "cpu"
	if useGPUifAvailable:
		print(language["warn_cudanotavailable"])

print("\033[H\033[2J\033[H")
print(language["info_loading"])

tokenizer = AutoTokenizer.from_pretrained(getSelectedModel())
if getSelectedModel() != "bigscience/T0_3B": # T0 uses different model generator
	model = AutoModelForCausalLM.from_pretrained(getSelectedModel(), torch_dtype=dataType, low_cpu_mem_usage=True).to(dev)
else:
	model = AutoModelForSeq2SeqLM.from_pretrained(getSelectedModel(), torch_dtype=dataType, low_cpu_mem_usage=True).to(dev)

print("\033[H\033[2J\033[H")

temp = 0.9 # temperature
topk = 50 # top k sampling
topp = 0.9 # top p sampling
length = 100 # max length of generated output
outcount = 1 # amount of generated outputs
lenp = 1.0 # length penalty
repp = 1.0 # repetition penalty

outputs = None
firstDone = False
wordLength = length
tokenLength = 0

while True:
	inText = input(language["input_prompt"])
	if inText.startswith("."):
		if inText.startswith(commands["readfile"]):
			splitted = re.split("(\.\w+) (\w+\.\w+)", inText)[2]
			try:
				with open(splitted, "r") as fl:
					inText = fl.read()
			except:
				print(language["error_unknownfile"])
				continue

			if len(inText) >= wordLength:
				wordLength = len(inText) + length
		elif inText.startswith(commands["readfile_additive"]):
			splitted = re.split("(\.\w+) (\w+\.\w+)", inText)[2]
			try:
				with open(splitted, "r") as fl:
					inText = fl.read()
			except:
				print(language["error_unknownfile"])
				continue

			inText += input(language["input_additional"])
			if len(inText) >= wordLength:
				wordLength = len(inText) + length
		elif inText.lower().startswith(commands["temperature"]):
			temp = float(re.split("(\.\w+) (\d+\.\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["length_penalty"]):
			lenp = float(re.split("(\.\w+) (\d+\.\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["repetition_penalty"]):
			repp = float(re.split("(\.\w+) (\d+\.\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["top_k_sampling"]):
			topk = int(re.split("(\.\w+) (\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["top_p_sampling"]):
			topp = float(re.split("(\.\w+) (\d+\.\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["wordlength"]):
			wordLength = int(re.split("(\.\w+) (\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["outputcount"]):
			outcount = int(re.split("(\.\w+) (\d+)", inText.lower())[2])
			continue
		elif inText.lower().startswith(commands["resethistory"]):
			print("\033[H\033[2J\033[H")
			wordLength = length
			tokenLength = 0
			outputs = None
			firstDone = False
			continue
		elif inText.lower().startswith(commands["exitscript"]):
			break
		elif inText.lower().startswith(commands["multilineinput"]):
			print(language["input_multiline"])
			while True:
				try:
					inText += input() + "\n"
				except EOFError:
					break
		else:
			print(language["error_unknowncommand"])
			continue
	else:
		if len(inText) >= wordLength:
			wordLength = len(inText) + length

	inputs = tokenizer.encode(inText + '\n', return_tensors="pt").to(dev)
	inputs = inputs.to(torch.int32)
	# TODO: Choose output for next input
	stripped_inputs = torch.cat([outputs if outcount == 1 else outputs[0], inputs], dim=-1).to(dev) if firstDone else inputs
	print(f"{language['info_newlength']} {tokenLength} + {len(stripped_inputs[0])}")
	tokenLength += len(stripped_inputs[0])
	with torch.cuda.amp.autocast(enabled=usingGPU):
		outputs = model.generate(
			stripped_inputs,
			do_sample=True,
			max_length=wordLength, top_p=topp, top_k=topk, temperature=temp,
			num_return_sequences=outcount, length_penalty=lenp, repetition_penalty=repp,
			pad_token_id=tokenizer.eos_token_id
		).to(dev)

	# T0 model gets rid of input by itself, while GPT-Neo and GPT-J do not
	if getSelectedModel() != "bigscience/T0_3B":
		wordLength += len(tokenizer.decode(outputs[0][stripped_inputs.shape[-1]:], skip_special_tokens=True))
		for i in range(len(outputs)):
			print("\n### " + language["info_output"] + f" {i+1} ###\n{tokenizer.decode(outputs[i][stripped_inputs.shape[-1]:], skip_special_tokens=True)}\n##################\n")
		print(f"{language['info_newlength']} {tokenLength} + {len(outputs[0][stripped_inputs.shape[-1]:])}")
		tokenLength += len(outputs[0][stripped_inputs.shape[-1]:])
	else:
		wordLength += len(tokenizer.decode(outputs[0], skip_special_tokens=True))
		for i in range(len(outputs)):
			print("\n### " + language["info_output"] + f" {i+1} ###\n{tokenizer.decode(outputs[i], skip_special_tokens=True)}\n##################\n")
		print(f"{language['info_newlength']} {tokenLength} + {len(outputs[0])}")
		tokenLength += len(outputs[0])

	firstDone = True