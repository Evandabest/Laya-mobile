"""Canonical mobile parity requests.

Keep these cases backend-neutral. Generated fixtures contain their fully expanded state and all
reference tensors so Swift and, later, Kotlin do not need to execute this module.
"""

from copy import deepcopy

BASE_STATE = {
    "from": "user@example.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": (
        "We were billed twice for March. Please refund the duplicate today or we will cancel "
        "our plan."
    ),
}

BASE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    },
}


def _case(name, state, questions=None):
    return {
        "name": name,
        "state": state,
        "questions": deepcopy(BASE_QUESTIONS if questions is None else questions),
    }


def parity_cases():
    """Representative requests whose expanded values are embedded in generated fixtures."""
    cases = [
        _case("typed_email", deepcopy(BASE_STATE)),
        _case("empty_state", ""),
        _case(
            "minimal_input",
            "x",
            {
                "binary_choice": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": ["a", "b"],
                }
            },
        ),
        _case(
            "single_option",
            "Only one outcome is permitted.",
            {
                "only": {
                    "type": "choice",
                    "instructions": "Choose the allowed outcome.",
                    "criteria": ["allowed"],
                }
            },
        ),
        _case(
            "unicode_mixed",
            {
                "message": "发票4411被重复扣款，请退款。 Café naïve — مرحباً 👩🏽‍💻🏳️‍🌈",
                "language": "中文 / Français / العربية",
            },
        ),
        _case(
            "whitespace_and_punctuation",
            "  tabs\tnewlines\n\nCRLF\r\nquotes ‘single’ “double” … !?  ",
        ),
        _case(
            "mask_literals",
            "[MASK] <mask> hello [MASK] <mask>",
            {
                "safe": {
                    "type": "noul",
                    "instructions": "Does [MASK] remain user text?",
                    "labels": {"false": "no", "true": "yes"},
                }
            },
        ),
        _case(
            "structured_values",
            deepcopy(BASE_STATE),
            {
                "choice": {
                    "type": "choice",
                    "instructions": {"task": "choose department", "priority": 2},
                    "criteria": {
                        "billing": {"description": "refunds", "code": 4},
                        "other": False,
                    },
                },
                "score": {
                    "type": "score",
                    "instructions": ["Estimate", "urgency"],
                    "criteria": [{"level": "low"}, 1, "high"],
                },
                "noul": {
                    "type": "noul",
                    "instructions": "Refund?",
                    "criteria": {"false": {"reason": "no request"}, "true": "money back"},
                },
            },
        ),
        _case(
            "conversation_tail",
            [
                {"role": "system", "content": "Old context that may be truncated."},
                {"role": "user", "content": "The latest request is a duplicate-charge refund."},
            ],
        ),
        _case(
            "many_options",
            deepcopy(BASE_STATE),
            {
                "department": {
                    "type": "choice",
                    "instructions": "Which department handles billing?",
                    "criteria": ["billing"] + [f"department_{index}" for index in range(19)],
                }
            },
        ),
        _case(
            "option_budget_pressure",
            "Choose the closest category.",
            {
                "category": {
                    "type": "choice",
                    "instructions": "Select one detailed category from the complete taxonomy.",
                    "criteria": {
                        f"category_{index}": (
                            "a deliberately long description sharing common prefix tokens "
                            f"with unique suffix {index}"
                        )
                        for index in range(40)
                    },
                }
            },
        ),
        _case("maximum_context_string", "token " * 2000),
        _case(
            "maximum_context_conversation",
            [
                {"role": "user", "content": "old " * 1000},
                {"role": "user", "content": "newest duplicate charge refund " * 300},
            ],
        ),
    ]

    multilingual = {
        "zh": "发票4411被重复扣款，请今天退款。",
        "de": "Ich wurde zweimal belastet. Bitte erstatten Sie den Betrag.",
        "hi": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
        "ja": "二重に請求されました。返金してください。",
        "ru": "С меня дважды списали деньги, верните деньги.",
    }
    cases.extend(
        _case(f"language_{lang}", {"message": text}) for lang, text in multilingual.items()
    )
    return cases
