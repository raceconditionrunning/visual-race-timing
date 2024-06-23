import sys
from typing import List, Tuple

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.validation import Validator, ValidationError


def ask_for_id(choices, show_default=False, allow_other=False) -> str:
    prompt_text = "Bib: " if not allow_other else f"Bib/(U)nknown: "
    # Fallback for non-tty host
    if not sys.stdin.isatty():
        for count, (bib, meta) in enumerate(choices, 1):
            print(f"{bib} {' '.join(map(str, meta))}".ljust(30), end='')
            if count % 4 == 0:
                print()
        print()
        return input(prompt_text).strip()

    default = choices[0][0] if choices else ''
    validator = HexNumberValidator() if not allow_other else HexNumberOrOtherValidator()
    session = PromptSession()

    def bib_to_name():
        # Get the current session text and check if it's a bib
        bib = session.default_buffer.text.strip()
        if bib in [choice[0] for choice in choices]:
            name = [choice[1][0] for choice in choices if choice[0] == bib][0]
            return f"{name}"
        return "Unknown"

    user_input = session.prompt(prompt_text, completer=MetadataCompleter(choices), validator=validator, default=default,
                                bottom_toolbar=bib_to_name)
    return user_input.strip()


class HexNumberValidator(Validator):
    def validate(self, document):
        text = document.text

        if not text:
            raise ValidationError(message='Please enter a number')
        try:
            int(text, 16)
        except ValueError:
            # Get the index of the first non-hex character
            index = 0
            for i, c in enumerate(text):
                if c not in '0123456789abcdefABCDEF':
                    index = i
                    break
            raise ValidationError(message='This input contains non-hex characters', cursor_position=index)


class HexNumberOrOtherValidator(HexNumberValidator):
    def validate(self, document):
        text = document.text
        if text.startswith('U') or text.startswith('u') or text.startswith('skip'):
            return
        super().validate(document)


class MetadataCompleter(Completer):
    def __init__(self, words_with_metadata: List[Tuple[str, str]]):
        # words_with_metadata is a list of tuples (word, metadata)
        self.words_with_metadata = words_with_metadata
        for word, metadata in self.words_with_metadata:
            if not isinstance(word, str):
                raise ValueError(f"Expected a string, got {type(word)} for word {word}")

    def get_completions(self, document, complete_event):
        word_before_cursor = document.get_word_before_cursor().lower()
        for word, metadata in self.words_with_metadata:
            if str(word).startswith(word_before_cursor):
                display_meta = f'{word} {" ".join(map(str, metadata))}'
                yield Completion(word, start_position=-len(word_before_cursor), display=display_meta)
            else:
                # Check if they've typed any metadata (e.g. runner name)
                for meta_element in metadata:
                    if meta_element.lower().startswith(word_before_cursor):
                        display_meta = f'{word} {" ".join(map(str, metadata))}'
                        yield Completion(word, start_position=-len(word_before_cursor), display=display_meta)
