import random

# Pre-defined romantic captions
CAPTIONS = [
    "This is how I see you on the days you laugh without trying.",
    "A gentle echo of your soul painted in color and light.",
    "If love had a canvas, this would be my first stroke.",
    "You, through the eyes of a dreamer.",
    "This isn’t just a style—it’s a feeling you leave behind.",
    "In every brushstroke, I find a piece of your heart.",
    "Your beauty transforms the ordinary into the extraordinary.",
    "I see your kindness in every vibrant hue.",
    "Art imitates life, but loving you defines it.",
    "My heart composes sonnets when it beholds your portrait.",
    "Every color whispers your name in the language of love.",
    "You are the masterpiece I never knew I could create.",
    "Your essence turns every shadow into light.",
    "In the gallery of my heart, you are the grandest exhibit.",
    "Your smile is the art I want to admire forever."
]

def get_random_caption() -> str:
    """
    Return a random pre-defined romantic caption.
    """
    return random.choice(CAPTIONS)

def generate_caption(prompt: str = None) -> str:
    """
    Placeholder for future GPT integration.
    Currently returns a random pre-defined caption.
    """
    return get_random_caption()
