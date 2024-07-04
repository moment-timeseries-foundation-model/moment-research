from transformers import T5Config, T5EncoderModel

model_config = T5Config.from_pretrained("t5-small")
print(model_config)

model = T5EncoderModel.from_pretrained("t5-small")
print(model)
