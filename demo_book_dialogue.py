import os
import sys
import time

# System path patch
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.qa_engine.book_dialogue_engine import HyperSymbolicBookDialogueEngine

def main():
    print("\n" + "=" * 80)
    print("HYPER-SYMBOLIC BOOK DIALOGUE ENGINE (BEZ LLM, 0 GRADIENT EPOCHS)")
    print("Książka: 'Mały Książę' (Antoine de Saint-Exupéry)")
    print("Author: Principal Software Architect")
    print("=" * 80)
    
    # 1. Struktura Wiedzy i Relacji Książki "Mały Książę"
    book_relations = [
        ("Mały Książę", "kocha i pielęgnuje", "Róża"),
        ("Róża", "jest dumna i wymaga opieki od", "Mały Książę"),
        ("Lis", "uczy oswojenia i mądrości", "Mały Książę"),
        ("Mały Książę", "oswaja", "Lis"),
        ("Pilot", "zaprzyjaźnia się z", "Mały Książę"),
        ("Wąż", "pomaga powrócić na planetę", "Mały Książę"),
        ("Mały Książę", "zamieszkuje", "Asteroida B612"),
        ("Mały Książę", "odwiedza", "Ziemia"),
        ("Król", "rozkazuje", "Mały Książę"),
        ("Latarnik", "zapala latarnię dla", "Mały Książę")
    ]
    
    chapter_concepts = {
        1: {"róża": 0.9, "miłość": 0.8},
        2: {"podróż": 0.7, "król": 0.6},
        3: {"lis": 0.95, "oswojenie": 1.0, "przyjaźń": 0.9},
        4: {"wąż": 0.85, "powrót": 0.9}
    }
    
    timeline = [
        "Planeta Asteroida B612 (Opieka nad Różą)",
        "Podróż po Planetach (Król, Próżny, Pijak, Latarnik)",
        "Lądowanie na Ziemi na pustyni",
        "Spotkanie z Lisem i Nauka Oswojenia",
        "Rozmowa z Pilotem i Rysunek Baranka",
        "Powrót do Róży za pomocą Węża"
    ]
    
    # 2. Inicjalizacja i Analityczne "Uczenie" Książki (0 Epok Gradientowych)
    engine = HyperSymbolicBookDialogueEngine(book_title="Mały Książę")
    t0 = time.perf_counter()
    fit_ms = engine.ingest_book_structure(book_relations, chapter_concepts, timeline)
    
    print(f"[+] Nauka Struktury Książki Ukończona w {fit_ms:.2f} ms! (0 epok gradientowych)")
    print(f"[+] Załadowano {len(book_relations)} relacji postaci oraz {len(timeline)} punktów chronologii.")
    print("=" * 80)
    print("\nPRZYKŁADOWE PYTANIA DO WPISANIA:")
    print(" - 'Kim jest Lis dla Małego Księcia?'")
    print(" - 'Co łączy Małego Księcia z Różą?'")
    print(" - 'Co wiesz o postaci Pilot?'")
    print(" - 'Jaka jest chronologia zdarzeń w książce?'")
    print(" - 'W których rozdziałach występuje motyw oswojenia?'")
    print(" - 'exit' / 'quit' (Zakończenie pracy CLI)")
    print("-" * 80)
    
    while True:
        try:
            user_input = input("\n[TWÓJ INPUT - PYTANIE O KSIĄŻKĘ] > ")
            cleaned = user_input.strip()
            
            if cleaned.lower() in ["exit", "quit", "q", "wyjście"]:
                print("[SYSTEM]: Zakończono sesję rozmowy o książce. Do widzenia!")
                break
                
            if not cleaned:
                continue
                
            # Odpowiedź w ułamku milisekundy BEZ LLM
            ans = engine.ask(cleaned)
            print(f"\n{ans}")
            
        except KeyboardInterrupt:
            print("\n[SYSTEM]: Zakończono sesję.")
            break

if __name__ == "__main__":
    main()
