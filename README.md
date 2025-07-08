# 🤟 Sistema de Reconhecimento de Gestos em Língua Gestual Portuguesa (LGP)

Este repositório apresenta um sistema desenvolvido para o reconhecimento de gestos dinâmicos em **Língua Gestual Portuguesa (LGP)**, com o objetivo de promover a inclusão social de pessoas surdas, facilitando a comunicação com ouvintes através da tradução de gestos em tempo real para texto.

O projeto foi desenvolvido por **Ivanilson Braga**, **Zakhar Khomyakivskyy** e **Ektiandro Elizabeth**, no âmbito da unidade curricular de Laboratorio De Projeto.

## 📋 Descrição Geral

A aplicação utiliza um modelo de Deep Learning para reconhecer gestos a partir de sequências de marcos (landmarks) da mão, extraídos com MediaPipe. Este método é altamente eficiente e ideal para integração em dispositivos móveis, permitindo o reconhecimento em tempo real com baixo custo computacional.

O sistema foi treinado para reconhecer os seguintes gestos:
- **água**
- **bom dia**
- **não**
- **olá**
- **por favor**
- **sim**

## 🛠 Tecnologias Utilizadas

- **Python 3.10+** – Linguagem principal de desenvolvimento
- **TensorFlow / Keras** – Construção e treino da rede neural LSTM
- **MediaPipe** – Extração de marcos da mão a partir de imagens
- **OpenCV** – Processamento de imagem
- **NumPy** – Manipulação de dados numéricos
- **Scikit-learn** – Divisão de dados para treino e validação
- **TensorFlow Lite** – Versão leve do modelo para dispositivos móveis

## 🧠 Como o modelo funciona?

O sistema abandonou a abordagem inicial de CNN (Rede Neural Convolucional) em favor de uma arquitetura mais moderna e eficiente para o reconhecimento de gestos dinâmicos: um **modelo LSTM (Long Short-Term Memory)**.

O fluxo de trabalho é o seguinte:
1.  **Extração de Marcos**: Em vez de usar imagens brutas, o script `processar_dados.py` utiliza o MediaPipe para analisar cada imagem do dataset e extrair as coordenadas 3D (x, y, z) dos 21 pontos-chave da mão. As imagens com desenhos de esqueleto são usadas apenas como fonte para gerar estes dados numéricos limpos.
2.  **Criação de Sequências**: Os marcos extraídos são agrupados em sequências de comprimento fixo (e.g., 20 frames). Estas sequências representam a trajetória do gesto ao longo do tempo.
3.  **Treino do Modelo LSTM**: O modelo LSTM é treinado com estas sequências. A sua arquitetura é especializada em aprender padrões temporais, tornando-o ideal para entender o movimento que define um gesto.
4.  **Conversão para TFLite**: O modelo treinado é convertido para o formato `.tflite`, que é otimizado para inferência rápida e de baixo consumo em dispositivos móveis como Android.

Esta abordagem é significativamente mais leve e rápida do que uma CNN, pois o modelo processa apenas algumas centenas de coordenadas em vez de dezenas de milhares de pixels por imagem.

## 📂 Estrutura do Projeto

```bash
📁 Reconhecimento_LGP/
│
├── processar_dados.py           # 1. Extrai marcos das imagens e cria o arquivo de dados.
├── treinar_modelo.py            # 2. Treina o modelo LSTM com os dados processados.
├── converter_tflite.py          # 3. Converte o modelo treinado para formato .tflite.
│
├── dados_processados.npz        # Arquivo com os dados de marcos, pronto para o treino.
├── modelo_gestos_lgp.h5         # Modelo treinado (Keras).
├── modelo_gestos_lgp.tflite     # Modelo convertido para Android.
├── classes.json                 # Mapeamento de gestos para IDs numéricos.
├── README.md                    # Este ficheiro.
└── dataset_limpo/               # Dataset de imagens originais.
```

## 🚀 Como Usar o Projeto

Execute os scripts na seguinte ordem para processar os dados, treinar o modelo e convertê-lo para uso móvel.

**1. Processar o Dataset de Imagens**
Este script analisa as imagens em `dataset_limpo/`, extrai os marcos da mão e cria o arquivo `dados_processados.npz`.
```bash
python processar_dados.py
```

**2. Treinar o Modelo LSTM**
Este script carrega os dados processados e treina o modelo LSTM, salvando o resultado como `modelo_gestos_lgp.h5`.
```bash
python treinar_modelo.py
```

**3. Converter para TensorFlow Lite**
Este script converte o modelo `.h5` para o formato `modelo_gestos_lgp.tflite`, que está pronto para ser usado na aplicação Android.
```bash
python converter_tflite.py
```

---
## 🤖 Aplicação Android

A versão móvel da aplicação, localizada na pasta `Reconhecimento_LGP_Android/`, foi desenvolvida em **Kotlin** com **Jetpack Compose** e utiliza os seguintes componentes:
- **CameraX**: Para uma implementação robusta da câmara.
- **MediaPipe (Tasks Vision)**: Para extrair os marcos da mão em tempo real.
- **TensorFlow Lite**: Para executar o modelo `modelo_gestos_lgp.tflite` e fazer a inferência.

O processamento é todo feito no dispositivo, usando a câmara para interpretar os gestos em tempo real com baixa latência.

### 🚀 Como Usar a Aplicação Android

1.  **Abrir no Android Studio**:
    *   Abra o Android Studio.
    *   Selecione `File > Open` e navegue até à pasta `Reconhecimento_LGP_Android`.
    *   Aguarde o **Gradle Sync** terminar. Este processo descarrega todas as dependências necessárias.

2.  **Selecionar o Dispositivo**:
    *   No topo da janela, escolha um dispositivo (físico ou emulador) no menu dropdown.
    *   Para usar um telemóvel físico, ative as "Opções de Programador" e a "Depuração USB".

3.  **Executar a Aplicação**:
    *   Clique no ícone verde de "Run 'app'" (▶️) ou use o atalho `Shift + F10`.
    *   A aplicação será instalada e iniciada no dispositivo selecionado.

> **Nota sobre a Versão do AGP**: O projeto foi configurado com a versão `8.4.1` do Android Gradle Plugin. Se o seu Android Studio sugerir uma atualização (ex: para `8.11`), pode aceitá-la. Desde que o Gradle Sync seja concluído com sucesso, a aplicação funcionará corretamente.

## 👥 Contribuidores

- **Ivanilson Braga**
- **Zakhar Khomyakivskyy**
- **Ektiandro Elizabeth**

## 🧾 Licença

Projeto desenvolvido para fins académicos no âmbito da unidade curricular de Projeto.
Qualquer reutilização parcial ou total deve referenciar os autores.

