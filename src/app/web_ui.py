import os
import sys
from nicegui import app, ui

# Add the repository root so the app can import modules from src/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.repetition.engine import get_due_reviews, update_review_state
from src.repetition.sm2 import calculate_sm2

# Serve generated sketch assets so Markdown images load in the browser.
assets_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/output_markdown/assets")
)
if os.path.exists(assets_path):
  app.add_static_files("/assets", assets_path)


@ui.page("/")
def review_dashboard():
  due_items = get_due_reviews(limit=20)
  current_index = 0

  # Build the main layout with Tailwind classes.
  with ui.column().classes(
      "w-full max-w-4xl mx-auto p-6 items-center min-h-screen bg-gray-50"
  ):
    # Header
    with ui.row().classes("w-full justify-between items-center mb-6"):
      with ui.column():
        ui.label("🧠 Knowledge Spaced Repetition").classes(
            "text-3xl font-extrabold text-gray-800"
        )
        ui.label(
            "Langfristige Wissenssicherung für Data Science & ML"
        ).classes("text-sm text-gray-500")
      ui.button(
          "Neu laden", on_click=lambda: refresh_data(), icon="refresh"
      ).classes("bg-white text-gray-700 border border-gray-300 shadow-sm")

    # Main chapter card.
    card_container = ui.card().classes(
        "w-full p-8 shadow-xl rounded-2xl bg-white border border-gray-100"
    )

    def render_current_card():
      card_container.clear()
      with card_container:
        if not due_items or current_index >= len(due_items):
          # Show the completion screen when no chapters are due.
          with ui.column().classes(
              "w-full items-center justify-center py-12 text-center"
          ):
            ui.icon("check_circle", color="green").classes("text-6xl mb-4")
            ui.label("Fantastisch! Alle Kapitel erledigt.").classes(
                "text-2xl font-bold text-gray-800 mb-2"
            )
            ui.label(
                "Du hast alle fälligen Kapitel für heute erfolgreich"
                " durchgearbeitet."
            ).classes("text-gray-500 mb-6")
            ui.button(
                "Erneut prüfen", on_click=lambda: refresh_data()
            ).classes(
                "bg-indigo-600 text-white px-6 py-2 rounded-xl shadow hover:bg-indigo-700"
            )
          return

        item = due_items[current_index]

        # Status and metadata bar.
        with ui.row().classes(
            "w-full justify-between items-center text-xs font-semibold text-gray-400 mb-4 pb-2 border-b"
        ):
          ui.label(
              f"LERN-EINHEIT #{current_index + 1} von {len(due_items)}"
          ).classes("uppercase tracking-wider")
          ui.label(
              f"{item['chapter_title']} | Wiederholungen:"
              f" {item['repetition_count']} | Intervall:"
              f" {item['interval_days']} Tage"
          )

        # Render chapter Markdown with LaTeX support and adjusted asset paths.
        content = item["content"]
        # Route relative asset paths through the FastAPI static server.
        content_fixed = content.replace("assets/", "/assets/")

        ui.label("Zusammenfassung & Herleitung:").classes(
            "font-bold text-gray-700 mb-2 text-sm"
        )
        with ui.element("div").classes(
            "w-full p-4 bg-gray-50 rounded-xl border border-gray-200"
        ):
          ui.markdown(
              content_fixed, extras=["latex", "fenced-code-blocks", "tables"]
          ).classes("w-full prose max-w-none text-gray-800 leading-relaxed")

        ui.separator().classes("my-6")

        # Rating controls using the SM-2 scale from 0 to 5.
        ui.label(
            "Wie gut konntest du dich an diesen Inhalt erinnern?"
        ).classes("font-semibold text-gray-700 text-center w-full mb-4")

        with ui.row().classes(
            "w-full justify-center gap-3 flex-wrap pb-2"
        ):
          ui.button(
              "0 - Komplett weg", on_click=lambda: handle_rating(0)
          ).classes(
              "bg-red-500 text-white font-medium px-4 py-2 rounded-xl"
              " shadow-md hover:bg-red-600 transition"
          )
          ui.button("1 - Sehr schwer", on_click=lambda: handle_rating(1)).classes(
              "bg-rose-500 text-white font-medium px-4 py-2 rounded-xl"
              " shadow-md hover:bg-rose-600 transition"
          )
          ui.button("2 - Schwer", on_click=lambda: handle_rating(2)).classes(
              "bg-orange-500 text-white font-medium px-4 py-2 rounded-xl"
              " shadow-md hover:bg-orange-600 transition"
          )
          ui.button("3 - Passabel", on_click=lambda: handle_rating(3)).classes(
              "bg-amber-500 text-white font-medium px-4 py-2 rounded-xl"
              " shadow-md hover:bg-amber-600 transition"
          )
          ui.button("4 - Gut", on_click=lambda: handle_rating(4)).classes(
              "bg-blue-600 text-white font-medium px-4 py-2 rounded-xl"
              " shadow-md hover:bg-blue-700 transition"
          )
          ui.button("5 - Perfekt", on_click=lambda: handle_rating(5)).classes(
              "bg-emerald-600 text-white font-medium px-4 py-2 rounded-xl"
              " shadow-md hover:bg-emerald-700 transition"
          )

    def handle_rating(score: int):
      nonlocal current_index
      if current_index >= len(due_items):
        return

      item = due_items[current_index]

      # Calculate the updated SM-2 schedule.
      new_rep, new_int, new_ef, next_date = calculate_sm2(
          quality=score,
          repetition_count=item["repetition_count"],
          interval_days=item["interval_days"],
          ease_factor=float(item["ease_factor"]),
      )

      # Persist the review state in PostgreSQL.
      update_review_state(
          review_id=item["review_id"],
          quality_score=score,
          new_rep=new_rep,
          new_interval=new_int,
          new_ef=new_ef,
          next_date=next_date,
      )

      # Show the next chapter.
      current_index += 1
      render_current_card()

    def refresh_data():
      nonlocal due_items, current_index
      due_items = get_due_reviews(limit=20)
      current_index = 0
      render_current_card()

    # Render the first chapter when the page loads.
    render_current_card()


if __name__ in {"__main__", "__mp_main__"}:
  ui.run(
      port=int(os.environ.get("APP_PORT", "8080")),
      title="Knowledge Spaced Repetition",
      dark=False,
  )