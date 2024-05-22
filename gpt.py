# pip install transformers SpeechRecognition pyttsx3 torch torchvision

import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import torch

recognizer = sr.Recognizer()
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def capturar_audio():
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        audio = recognizer.listen(mic)
        try:
            text = recognizer.recognize_google(audio, language='pt')
            return text.lower()
        except sr.UnknownValueError:
            return "Não entendi o que foi dito"
        except sr.RequestError:
            return "Erro ao se comunicar com o serviço de reconhecimento de fala"

texto_reconhecido = capturar_audio()

qea = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese", framework="pt")

texto = """
Buracos negros são regiões do espaço onde a gravidade é tão intensa que nada, nem mesmo a luz, pode escapar. Eles se formam quando uma quantidade suficiente de matéria é compactada em um espaço pequeno, geralmente como resultado do colapso gravitacional de uma estrela massiva ao final de sua vida. Esse processo cria uma singularidade, um ponto de densidade infinita e volume zero, onde as leis da física, como as conhecemos, deixam de se aplicar. Ao redor da singularidade existe o horizonte de eventos, a fronteira além da qual nada pode escapar da atração gravitacional do buraco negro. Tudo que passa pelo horizonte de eventos está irremediavelmente perdido para o resto do universo. Devido à sua natureza, buracos negros não podem ser observados diretamente. No entanto, eles podem ser detectados pela influência que exercem sobre a matéria e a luz ao seu redor. Por exemplo, a matéria que cai em um buraco negro pode formar um disco de acreção, aquecendo-se e emitindo radiação intensa antes de ser engolida. Existem diferentes tipos de buracos negros, classificados de acordo com sua massa. Os buracos negros estelares, formados a partir do colapso de estrelas massivas, têm massas entre 3 e 20 vezes a massa do Sol. Buracos negros supermassivos, com massas que variam de milhões a bilhões de vezes a massa solar, são encontrados nos centros de galáxias, incluindo a nossa Via Láctea. A origem exata dos buracos negros supermassivos ainda é um tópico de pesquisa ativa. Além disso, a existência de buracos negros de massa intermediária, com massas entre 100 e 1000 vezes a massa do Sol, tem sido proposta, embora menos comuns e mais difíceis de detectar. Buracos negros primordiais, hipotéticos buracos negros que teriam se formado no início do universo, também são uma área de investigação teórica. Buracos negros desempenham um papel crucial na dinâmica e evolução das galáxias. Eles podem influenciar a formação de estrelas e a distribuição de matéria no espaço. Estudos de buracos negros também oferecem uma oportunidade única para testar a teoria da relatividade geral em condições extremas, avançando nossa compreensão da física fundamental. A detecção de ondas gravitacionais, ondulações no tecido do espaço-tempo causadas por eventos cataclísmicos como a fusão de buracos negros, abriu uma nova janela para a observação do universo. As primeiras detecções dessas ondas, feitas pelo Observatório de Ondas Gravitacionais por Interferômetro Laser (LIGO) em 2015, confirmaram a existência de buracos negros binários e proporcionaram uma nova maneira de estudar essas misteriosas entidades cósmicas. Em suma, buracos negros são objetos astronômicos fascinantes que continuam a desafiar e expandir nosso entendimento do universo. Estudos contínuos e futuras descobertas prometem revelar ainda mais sobre esses misteriosos habitantes do cosmos.
"""

pergunta = texto_reconhecido

resposta = qea(question=pergunta, context=texto)

print("Pergunta: ", pergunta)
print("Resposta: ", resposta['answer'])
print("Score: ", resposta['score'])

engine.setProperty('voice', voices[2].id)
engine.setProperty('rate', 190)

engine.say(resposta['answer'])
engine.runAndWait()
