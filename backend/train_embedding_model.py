from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

train_examples = []

def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            train_examples.append(
                InputExample(
                    texts=[item["query"], item["passage"]],
                    label=1.0
                )
            )

load_pairs("hadith_pairs.jsonl")
load_pairs("quran_pairs.jsonl")

print("Total training pairs:", len(train_examples))

train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=500,
    show_progress_bar=True
)

model.save("trained_islamic_embedding_model")
print("Model saved!")
