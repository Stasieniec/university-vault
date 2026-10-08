#!/usr/bin/env python3
"""Generate each course's home page and flashcards page from the notes themselves.

For every folder under Courses/ that has a "<COURSE> - Overview.md", this writes:

- Courses/<COURSE>/index.md
    The course home. On the site it is the folder page (/Courses/<COURSE>/): exam date,
    a table of lecture notes with what each covers and how many flashcards it has, the
    other notes, and the remaining folders (book notes, exercises, ...).
- Courses/<COURSE>/<COURSE> - Flashcards.md   (only when the course has flashcards)
    One section per note that has a "## Flashcards" section, transcluding it. The site's
    study mode turns this into "study everything / one lecture / any mix".
- Courses/<COURSE>/<COURSE> - Exam Questions.md   (only when the course has exam questions)
    The same, for each note's "## Exam questions" section: the long-form exam-style questions
    with key points and model answers, for written practice.

Both files are generated: edit this script, not the output. Re-run it after adding or
changing flashcards or lecture notes. It only rewrites a file when its content changed.

    python3 docs/superpowers/tools/course_pages.py
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

VAULT = Path(__file__).resolve().parents[3]
COURSES = VAULT / "Courses"

CARD_RE = re.compile(r"^> \[!(card|exam)\]-", re.MULTILINE)
FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")

# Subfolders listed on the course home, in this order, after the lectures.
SECTIONS = ["Book Notes", "Exercises", "Coding Assignments", "Exam Prep"]
SKIP_DIRS = {"Lectures", "Assets", "Assignments"}


def natural(p: Path) -> list:
    """Sort key so that Ch2 comes before Ch10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def nice_date(iso: str) -> str:
    try:
        return date.fromisoformat(iso[:10]).strftime("%a %d %b %Y")
    except ValueError:
        return iso


def frontmatter(text: str) -> str:
    m = FM_RE.match(text)
    return m.group(1) if m else ""


def fm_scalar(fm: str, key: str) -> str | None:
    m = re.search(rf"^{key}\s*:\s*(.+)$", fm, re.MULTILINE)
    return m.group(1).strip().strip("\"'") if m else None


def fm_list(fm: str, key: str) -> list[str]:
    m = re.search(rf"^{key}\s*:\s*\n((?:\s+-\s+.*\n?)+)", fm, re.MULTILINE)
    if not m:
        return []
    items = [re.sub(r"^\s+-\s+", "", line).strip().strip("\"'") for line in m.group(1).splitlines()]
    return [i for i in items if i]


def plain(text: str) -> str:
    """Wikilinks to their display text, so topic lists read as plain words."""
    return WIKILINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)


def flashcards_section(text: str) -> str | None:
    m = re.search(r"^## Flashcards\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1) if m else None


def exam_section(text: str) -> str | None:
    m = re.search(r"^## Exam questions\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1) if m else None


def card_counts(text: str) -> tuple[int, int]:
    """(exam questions, flashcards). Exam questions live in their own section."""
    cards = CARD_RE.findall(flashcards_section(text) or "")
    exams = CARD_RE.findall(exam_section(text) or "")
    return exams.count("exam") + cards.count("exam"), cards.count("card")


def split_name(stem: str, code: str) -> tuple[str, str]:
    """'PhilSci-L01b - Popper and Lakatos' -> ('L01b', 'Popper and Lakatos')."""
    head, _, title = stem.partition(" - ")
    label = head[len(code) + 1 :] if head.startswith(code + "-") else head
    return label, title or stem


def course_title(overview_text: str, code: str) -> str:
    m = re.search(r"^# (.+)$", overview_text, re.MULTILINE)
    if not m:
        return code
    return re.sub(r"\s*(—|-)\s*Overview\s*$", "", m.group(1)).strip()


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build(course_dir: Path) -> list[Path]:
    code = course_dir.name
    overview = course_dir / f"{code} - Overview.md"
    if not overview.exists():
        return []
    ov_text = overview.read_text(encoding="utf-8")
    ov_fm = frontmatter(ov_text)
    title = course_title(ov_text, code)
    course_code = fm_scalar(ov_fm, "course") or code
    exam = fm_scalar(ov_fm, "exam_date")

    lectures = sorted((course_dir / "Lectures").glob("*.md"), key=natural) if (course_dir / "Lectures").is_dir() else []
    generated = {"index", f"{code} - Overview", f"{code} - Flashcards", f"{code} - Exam Questions"}
    others = sorted((p for p in course_dir.glob("*.md") if p.stem not in generated), key=natural)

    decks: list[tuple[str, str, int, int]] = []  # (stem, heading, exam, recall)
    rows = []
    for p in lectures:
        text = p.read_text(encoding="utf-8")
        label, name = split_name(p.stem, code)
        topics = [plain(t) for t in fm_list(frontmatter(text), "topics")]
        covers = ", ".join(topics[:4]) + (", ..." if len(topics) > 4 else "")
        n_exam, n_card = card_counts(text)
        if n_exam + n_card:
            decks.append((p.stem, f"{label} {name}", n_exam, n_card))
        rows.append((label, p.stem, name, covers, n_card))

    other_rows = []
    for p in others:
        text = p.read_text(encoding="utf-8")
        n_exam, n_card = card_counts(text)
        name = p.stem[len(code) + 3 :] if p.stem.startswith(code + " - ") else p.stem
        if n_exam + n_card:
            decks.append((p.stem, name, n_exam, n_card))
        other_rows.append((p.stem, name, n_card))

    total = sum(c for _, _, _, c in decks)
    total_exam = sum(e for _, _, e, _ in decks)
    flash_stem = f"{code} - Flashcards"
    exam_stem = f"{code} - Exam Questions"
    out: list[Path] = []

    # Course home ---------------------------------------------------------------------
    lines = [
        "---",
        f'title: "{title}"',
        "type: moc",
        f"course: {course_code}",
        "tags: [moc, course-home]",
        "cssclasses: [course-home]",
        "status: complete",
        "---",
        "",
        "<!-- Generated by docs/superpowers/tools/course_pages.py. Edit the script, not this file. -->",
        "",
    ]
    facts = []
    if exam:
        facts.append(f"**Exam:** {nice_date(exam)}{' (past)' if exam[:10] < date.today().isoformat() else ''}")
    facts.append(f"**Lecture notes:** {len(lectures)}")
    if total:
        facts.append(f"**Flashcards:** {total}")
    if total_exam:
        facts.append(f"**Exam questions:** {total_exam}")
    lines += [" · ".join(facts), ""]
    lines += [f"Schedule, assessment and deadlines live in the [[{code} - Overview|course overview]].", ""]

    if total:
        lines += [
            "## Flashcards",
            "",
            f"{total} short single-fact cards across {sum(1 for d in decks if d[3])} sets. [[{flash_stem}|Open the flashcards]] to study the whole course, one lecture, or any mix of lectures. Each lecture note also ends with its own set.",
            "",
        ]
    if total_exam:
        lines += [
            f"[[{exam_stem}|Exam questions]]: {total_exam} long-form exam-style questions with key points and model answers, for written practice.",
            "",
        ]

    if rows:
        # One list item per lecture: number, title and card count on the first line, topics on
        # a second line in italics (the site styles that line small and grey). A table was
        # tried first and was unreadable on a phone.
        lines += ["## Lecture notes", ""]
        for label, stem, name, covers, n in rows:
            first = f"- **{label}** [[{stem}|{name}]]" + (f" · {n} cards" if n else "")
            lines.append(first + ("  " if covers else ""))
            if covers:
                lines.append(f"  *{covers}*")
        lines.append("")

    if other_rows:
        lines += ["## Other notes", ""]
        lines.append(f"- [[{code} - Overview|Overview]]: schedule, assessment, deadlines")
        for stem, name, n in other_rows:
            lines.append(f"- [[{stem}|{name}]]" + (f" ({n} flashcards)" if n else ""))
        lines.append("")

    for section in SECTIONS:
        d = course_dir / section
        if not d.is_dir():
            continue
        notes = sorted(d.glob("*.md"), key=natural)
        if not notes:
            continue
        lines += [f"## {section}", ""]
        for p in notes:
            label, name = split_name(p.stem, code)
            shown = f"{label}: {name}" if name != p.stem else p.stem
            lines.append(f"- [[{p.stem}|{shown}]]")
        lines.append("")

    extra_dirs = sorted(
        d.name for d in course_dir.iterdir() if d.is_dir() and d.name not in SKIP_DIRS and d.name not in SECTIONS
    )
    if extra_dirs:
        lines += ["## More", ""]
        for name in extra_dirs:
            first = next(iter(sorted((course_dir / name).glob("*.md"))), None)
            if first:
                lines.append(f"- {name}: [[{first.stem}|{first.stem}]] and the rest of the folder")
        lines.append("")

    home = course_dir / "index.md"
    if write_if_changed(home, "\n".join(lines).rstrip() + "\n"):
        out.append(home)

    # Flashcards page -----------------------------------------------------------------
    if total:
        fl = [
            "---",
            "type: flashcards",
            f"course: {course_code}",
            "tags: [flashcards]",
            "status: complete",
            "---",
            "",
            "<!-- Generated by docs/superpowers/tools/course_pages.py. Edit the script, not this file. -->",
            "",
            f"# {code} - Flashcards",
            "",
            f"All {total} flashcards for {title}, one set per note. Each card asks one thing and has a short answer. Say the answer before you reveal it, and only press Got it if you had all of it. Study everything at once, a single lecture, or tick any mix. Progress lives only in this browser tab and disappears when you close it. The long-form questions are on the [[{exam_stem}|exam questions]] page.",
            "",
            # Full path: every course has an "index", so a bare [[index]] is ambiguous.
            f"Back to the [[Courses/{code}/index|{title} home]].",
            "",
        ]
        for stem, heading, _, n_card in decks:
            if n_card:
                fl += [f"## {heading}", "", f"![[{stem}#Flashcards]]", ""]
        page = course_dir / f"{flash_stem}.md"
        if write_if_changed(page, "\n".join(fl).rstrip() + "\n"):
            out.append(page)

    # Exam questions page -------------------------------------------------------------
    if total_exam:
        ex = [
            "---",
            "type: flashcards",
            f"course: {course_code}",
            "tags: [flashcards, exam-prep]",
            "status: complete",
            "---",
            "",
            "<!-- Generated by docs/superpowers/tools/course_pages.py. Edit the script, not this file. -->",
            "",
            f"# {code} - Exam Questions",
            "",
            f"All {total_exam} exam-style questions for {title}, one set per note. Each answer opens with the key points a grader looks for, then a full model answer. Write or say your answer before revealing. The short drilling cards are on the [[{flash_stem}|flashcards]] page.",
            "",
            f"Back to the [[Courses/{code}/index|{title} home]].",
            "",
        ]
        for stem, heading, n_exam, _ in decks:
            if n_exam:
                ex += [f"## {heading}", "", f"![[{stem}#Exam questions]]", ""]
        page = course_dir / f"{exam_stem}.md"
        if write_if_changed(page, "\n".join(ex).rstrip() + "\n"):
            out.append(page)

    return out


def main() -> None:
    changed = []
    for d in sorted(COURSES.iterdir()):
        if d.is_dir():
            changed += build(d)
    for p in changed:
        print(f"updated {p.relative_to(VAULT)}")
    if not changed:
        print("course pages up to date")


if __name__ == "__main__":
    main()
