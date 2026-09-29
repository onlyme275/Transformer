import torch
import torch.nn as nn
from config import get_config
from model import build_transformer
from tokenizers import Tokenizer


# -----------------------------
# Configuration
# -----------------------------
config = get_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------
# Load tokenizers
# -----------------------------
tokenizer_src = Tokenizer.from_file(
    config["tokenizer_file"].format(config["lang_src"])
)

tokenizer_tgt = Tokenizer.from_file(
    config["tokenizer_file"].format(config["lang_tgt"])
)


# -----------------------------
# Build model
# -----------------------------
model = build_transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    config["seq_len"],
    config["seq_len"],
    d_model=config["d_model"],
    N=6,
    h=8,
    dropout=0.1,
    d_ff=2048,
)

# -----------------------------
# Load checkpoint
# -----------------------------
checkpoint = torch.load(
    "weights/tmodel_04.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded!")
print("Type an English sentence.")
print("Type 'quit' to exit.\n")


# -----------------------------
# Get special tokens
# -----------------------------
sos_token = tokenizer_tgt.token_to_id("[SOS]")
eos_token = tokenizer_tgt.token_to_id("[EOS]")
pad_token = tokenizer_tgt.token_to_id("[PAD]")


# -----------------------------
# Greedy decoding
# -----------------------------
def translate(sentence):

    # Encode English sentence
    source_tokens = tokenizer_src.encode(sentence).ids

    # Convert to tensor
    encoder_input = torch.tensor(
        source_tokens,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    # Source mask
    encoder_mask = (
        (encoder_input != pad_token)
        .unsqueeze(1)
        .unsqueeze(1)
        .int()
        .to(device)
    )

    # Encoder
    with torch.no_grad():
        encoder_output = model.encode(
            encoder_input,
            encoder_mask
        )

    # Start decoder with SOS
    decoder_input = torch.tensor(
        [[sos_token]],
        dtype=torch.long
    ).to(device)

    # Generate one token at a time
    for _ in range(config["seq_len"]):

        decoder_mask = torch.triu(
            torch.ones(
                (1, decoder_input.size(1), decoder_input.size(1)),
                device=device
            ),
            diagonal=1
        ).bool()

        with torch.no_grad():

            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask
            )

            probabilities = model.project(decoder_output)

        # Get last token
        next_token = torch.argmax(
            probabilities[:, -1, :],
            dim=-1
        )

        # Add token to decoder input
        decoder_input = torch.cat(
            [
                decoder_input,
                next_token.unsqueeze(0)
            ],
            dim=1
        )

        # Stop when EOS appears
        if next_token.item() == eos_token:
            break

    # Convert token IDs back to Italian text
    output_tokens = decoder_input.squeeze(0).tolist()

    # Remove SOS and EOS
    output_tokens = [
        token
        for token in output_tokens
        if token not in [sos_token, eos_token, pad_token]
    ]

    return tokenizer_tgt.decode(output_tokens)


# -----------------------------
# Interactive loop
# -----------------------------
while True:

    sentence = input("You: ")

    if sentence.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if not sentence.strip():
        continue

    translation = translate(sentence)

    print("Italian:", translation)
    print()


    