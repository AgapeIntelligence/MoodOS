import os
import json
import torch
from typing import Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

# Initialize client once (thread-safe if you avoid stateful usage)
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)


def moodvector_to_valence_arousal(
    mv: torch.Tensor,
    model: str = "grok-beta",  # or "grok-3" when available
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Convert a 32-dimensional MoodVector into DEAP-style valence and arousal scores (1–9)
    using Grok via xAI API.

    Returns a clean dictionary with:
        - valence (float 1.0–9.0)
        - arousal (float 1.0–9.0)
        - reasoning (str)

    Raises:
        ValueError: If API response cannot be parsed as JSON or values are out of range
        RuntimeError: If API call fails after retries
    """
    if mv.dim() != 1 or mv.shape[0] != 32:
        raise ValueError("Input must be a 1D tensor of length 32")

    # Convert to list with clean formatting (4 decimal places)
    mv_list = mv.tolist()
    mv_str = "[" + ", ".join(f"{x:.4f}" for x in mv_list) + "]"

    prompt = f"""You are an expert in affective computing.
Given this 32-dimensional continuous mood embedding (MoodVector32), predict the corresponding
valence and arousal values on the DEAP dataset scale (1 = low, 9 = high).

MoodVector: {mv_str}

Respond **only** with valid JSON in this exact format:
{{
  "valence": 7.2,
  "arousal": 6.8,
  "reasoning": "Brief explanation of the prediction (1-2 sentences)"
}}

Do not include any other text, markdown, or code blocks."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200,
            )

            content: str = response.choices[0].message.content.strip()

            # Handle common markdown wrappers
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)

            # Validate required fields and ranges
            valence = float(result["valence"])
            arousal = float(result["arousal"])

            if not (1.0 <= valence <= 9.0 and 1.0 <= arousal <= 9.0):
                raise ValueError("Valence/arousal out of range [1.0, 9.0]")

            return {
                "valence": valence,
                "arousal": arousal,
                "reasoning": result.get("reasoning", result.get("reason", "")).strip(),
                "raw_response": content,
            }

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to parse valid valence/arousal after {max_retries} attempts: {e}\nLast response: {content if 'content' in locals() else 'N/A'}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"xAI API error: {e}")

    return {"valence": 5.0, "arousal": 5.0, "reasoning": "Fallback due to API failure"}  # Unreachable


# ————————————————————————
# Example usage
# ————————————————————————
if __name__ == "__main__":
    # Dummy vector (replace with real MoodProcessor output)
    dummy_mv = torch.randn(32).tanh()  # bounded in [-1, 1]

    try:
        va = moodvector_to_valence_arousal(dummy_mv, temperature=0.0)
        print("Valence:", va["valence"])
        print("Arousal:", va["arousal"])
        print("Reasoning:", va["reasoning"])
    except Exception as e:
        print("Error:", e)
