# Wyoming gateway for streaming asr [model](https://docs.mistral.ai/models/voxtral-mini-transcribe-realtime-26-02) using Mistral API (paid).

Typical processing time is 0.1s and does not depend on the phrase length.

The model automatically determines the language from the available ones:

[Arabic	German	English	Spanish	French	Hindi	Italian	Dutch	Portuguese	Chinese	Japanese	Korean	Russian]


```
git clone https://github.com/mitrokun/wyoming_voxtral_api.git
cd wyoming_voxtral_api

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install "mistralai[realtime]" wyoming numpy

python3 -m asr_mistral \
    --api-key "your_key" \
    --uri "tcp://0.0.0.0:10300"
```
If you don't specify a port, the server will start on `10304`

The project can be easily migrated to use a [local model](https://huggingface.co/docs/transformers/main/en/model_doc/voxtral_realtime#streaming-transcription)


