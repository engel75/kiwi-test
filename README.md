# KV-Cache A/B-Test — Dokumentation zu `kiwi-test.sh`

## TL;DR

Der KV-Cache ist das "Arbeitsgedächtnis" eines LLM-Inferenzservers: Für jedes Token des Eingabetextes werden dort Zwischenwerte abgelegt, auf die das Modell bei jeder weiteren Berechnung zurückgreift. Um größere Kontextfenster in den GPU-Speicher zu bekommen, wird dieser Cache häufig von BF16 (16 Bit pro Wert) auf FP8 (8 Bit pro Wert) komprimiert — eine minimal verlustbehaftete Kompression, die die Denk- und Schlussfolgerungsqualität des Modells, in extrem seltenen Fällen beeinträchtigen *kann*, aber nicht muss. Dieses Script macht die Frage messbar: Es stellt dem Modell fünf automatisch generierte Aufgaben mit eindeutig feststehender Lösung — darunter genau die Aufgabentypen, die auf KV-Quantisierungsfehler am empfindlichsten reagieren (Zustandsverfolgung, Zahlenaggregation und mehrstufiges Schlussfolgern über sehr lange Dokumente). Weil die Aufgaben aus einem Zufallsgenerator mit festem Startwert (Seed) erzeugt werden, sind sie exakt reproduzierbar: Derselbe Seed erzeugt bit-identische Aufgaben. Dadurch lassen sich zwei Serverkonfigurationen — etwa BF16-KV gegen FP8-KV — mit **identischen** Eingaben vergleichen; der einzige Unterschied zwischen den Läufen ist die zu prüfende Konfiguration. Schneiden beide gleich ab, ist belegt, dass die Kompression die Denkleistung nicht messbar verschlechtert. Schneidet eine Konfiguration schlechter ab, zeigt der Test genau, bei welchem Aufgabentyp und ab welcher Kontextlänge es kippt. Alle Ergebnisse landen zusätzlich in einer CSV-Datei zur Auswertung.

---

## 1. Hintergrund: Was ist der KV-Cache und warum kann seine Präzision die Qualität beeinflussen?

Ein LLM speichert während der Verarbeitung für jedes Token des Eingabetextes zwei Vektoren im sogenannten **KV-Cache** (Key/Value-Cache). Bei jedem neu erzeugten Token schaut das Modell über den Attention-Mechanismus auf diese gespeicherten Vektoren zurück — der KV-Cache ist damit das Arbeitsgedächtnis des Modells über den gesamten Kontext.

Dieses Arbeitsgedächtnis ist groß: Bei mehreren hunderttausend Token Kontext belegt es viele Gigabyte GPU-Speicher, und seine Größe wächst linear mit der Kontextlänge. Wer das Kontextfenster vergrößern will, ohne mehr Hardware zu beschaffen, halbiert deshalb oft die Ablagepräzision — von **BF16** (16 Bit pro Wert) auf **FP8** (8 Bit pro Wert; die verbreitete Variante e4m3 hat nur 3 Bit Mantisse). Das ist eine verlustbehaftete Kompression: Jeder gespeicherte Wert wird leicht gerundet. Die Modellgewichte selbst sind davon nicht betroffen — aber jede Attention-Berechnung rechnet fortan mit minimal verrauschten Erinnerungen.

In der Praxis ist dieser Rundungsfehler fast immer unauffällig. Der Effekt wächst tendenziell mit der Kontextlänge, weil sich kleine Fehler über mehr gespeicherte Token akkumulieren. Ob und wie stark eine konkrete Kombination aus Modell, Serving-Framework und KV-Datentyp betroffen ist, lässt sich nicht pauschal beantworten — aber messen. Genau das tut dieses Script.

### Eine Analogie: MP3-Kompression eines Musikstücks

Wer den Schritt von BF16 auf FP8 greifbar machen will, kann ihn mit der MP3-Kompression eines Musikstücks vergleichen: eine verlustbehaftete Kompression einer Zwischenrepräsentation, die den Speicherbedarf halbiert — und deren Verluste, wenn überhaupt, zuerst bei "schwierigem Material" auftreten. Wichtig ist dabei die richtige Bitrate der Analogie: FP8-KV entspricht **nicht** dem Sprung von 320 auf 128 kbps (dort hört man Artefakte deutlich), sondern eher dem Schritt von 320 auf gute 192 kbps — **für das meiste, praktisch fast alles Material ist das Ergebnis vom Original nicht zu unterscheiden.** Messbar wird der Unterschied erst im gezielten "Blindtest" auf besonders schwierigem Material — und genau solche Blindtests sind die Tests T1–T3 dieses Scripts: Sie hören nicht nach Gefühl hin, sondern prüfen unter Laborbedingungen die härtesten Passagen.

An drei Stellen hinkt die Analogie allerdings, und die sind es wert, mitgedacht zu werden.

**Die Verlustart ist anders.** MP3 ist psychoakustisch *schlau*: Es wirft gezielt weg, was Menschen nachweislich nicht hören. FP8 ist dagegen gleichförmiges Runden ohne jedes Wahrnehmungsmodell — pro gespeichertem Wert sogar brutaler, als das 2:1-Speicherverhältnis suggeriert (e4m3 hat nur 3 Mantissen-Bits, BF16 hat 8). Dass es trotzdem so gut funktioniert, liegt nicht an cleverer Kompression, sondern daran, dass Attention über tausende Werte mittelt und sich Rundungsrauschen dabei weitgehend heraushebt.

**Das Degradationsverhalten ist anders.** MP3 verschlechtert kontinuierlich: Bei niedriger Bitrate klingt *alles* etwas matter. KV-Quantisierung verschlechtert diskret: Die Antwort ist nicht "10 % unschärfer formuliert", sondern in fast allen Fällen identisch gut — und in seltenen Fällen kippt eine Ziffer oder ein Schlussschritt, und die Antwort ist schlicht falsch. Es ist weniger "dumpferer Sound" als "sehr leicht erhöhte Fehlerwahrscheinlichkeit". Deshalb misst man es auch nicht durch Anhören einer einzelnen Antwort, sondern über Pass-Raten vieler unabhängiger Durchläufe — der Grund für den Seed-Ansatz in Kapitel 3.

**Der Ort des Verlusts ist anders.** Bei MP3 ist das komprimierte Signal direkt das, was man konsumiert. Der KV-Cache ist dagegen nur das *Arbeitsgedächtnis* — der Output kann trotz leicht verrauschtem Speicher perfekt sein. Passender als "die MP3, die man hört" wäre daher: Ein Tontechniker mischt mit leicht komprimierten Zuspielern ab — solange das Rauschen unter seiner Entscheidungsschwelle bleibt, ist das fertige Master davon nicht zu unterscheiden.

## 2. Die fünf Tests im Detail

Alle Langkontext-Tests (T1–T3) funktionieren nach demselben Prinzip: Das Script erzeugt ein langes "internes Dokument" aus unverfänglichen Fülltext-Sätzen und verteilt darin gezielt Fakten, deren korrekte Verarbeitung eine eindeutige, automatisch prüfbare Antwort ergibt. Das Modell kennt die Antwort nicht aus dem Training — sie ist zufällig generiert und existiert nur in diesem Dokument.

### T1 — Variable-Tracking (Zustandsverfolgung)

Im Dokument verteilt stehen zwölf Zeilen der Form `[SYSTEMLOG] Parameter ALPHA wurde auf den Wert 858 gesetzt.` — jedes Mal mit einem anderen Zufallswert. Gefragt ist der **letzte** zugewiesene Wert. Das Modell muss also alle zwölf Fundstellen erkennen, ihre Reihenfolge im Dokument korrekt erfassen und elf davon aktiv verwerfen. Reines Retrieval reicht hier nicht: Wer nur "irgendeinen" ALPHA-Wert findet, hat mit über 90 % Wahrscheinlichkeit den falschen. Genau diese Art von Positions- und Zustandspräzision leidet zuerst, wenn Attention-Werte durch Quantisierungsrauschen unscharf werden — T1 ist damit der empfindlichste Frühindikator im Testset.

### T2 — Aggregation (Informationen zusammentragen und verrechnen)

Fünfzehn Buchungszeilen mit zufälligen CHF-Beträgen sind über das gesamte Dokument verstreut; gefragt ist die Gesamtsumme. Das Modell muss ausnahmslos **alle** fünfzehn Stellen finden (eine übersehene Buchung → falsche Summe), die Beträge korrekt extrahieren und fehlerfrei addieren. Aggregation gilt in der Langkontext-Forschung als die härteste Standardkategorie, weil sich hier Retrieval-Vollständigkeit und Rechenpräzision multiplizieren. Eine Konfiguration, deren Langkontext-Verarbeitung degradiert ist, fällt hier zuverlässig auf.

### T3 — Multi-Hop (mehrstufige Schlusskette)

Drei Notizen an verschiedenen, zufälligen Stellen im Dokument: Der Serverschlüssel liegt bei Person X → Person X arbeitet an Standort Y → der Tresorcode an Standort Y lautet Z. Gefragt ist der Code, der nötig ist, um an den Schlüssel zu gelangen. Keine der drei Notizen beantwortet die Frage allein; das Modell muss die Kette Person → Ort → Code aktiv konstruieren. Das ist die Kernkompetenz hinter "Mitdenken": verteiltes Wissen logisch verknüpfen statt nur zitieren.

### T4 — Reasoning-Sanity (Kurzkontext-Baseline)

Eine klassische Altersrätsel-Aufgabe **ohne** langes Dokument. Da der KV-Cache bei winzigem Kontext praktisch keine Rolle spielt, muss dieser Test in jeder Konfiguration bestehen. Er dient als Kontrollgruppe: Fällt T4 durch, liegt das Problem nicht an der KV-Cache-Präzision, sondern woanders (Modell, Template, Sampling, Gateway) — und die Läufe T1–T3 wären erst nach Klärung aussagekräftig.

### T5 — Thinking-Check (ist der Denkprozess überhaupt aktiv?)

Eine simple Rechenaufgabe mit der Aufforderung, Schritt für Schritt zu denken. Bestanden ist der Test nur, wenn die Antwort korrekt ist **und** die API sichtbaren Reasoning-Output zurückliefert (`reasoning_content`, `reasoning` oder `reasoning_details` — je nach Server/Gateway-Format). Damit wird die banalste, aber häufigste Erklärung für gefühlte Qualitätsverluste geprüft: dass der Thinking-Modus oder der Reasoning-Parser in der Serverkonfiguration deaktiviert oder defekt ist. Das hätte mit der KV-Cache-Präzision nichts zu tun, würde sich für Nutzer aber exakt wie ein Qualitätsverlust anfühlen.

## 3. Seeds: Warum die Ergebnisse reproduzierbar und trotzdem vielfältig sind

Die Testdokumente werden von einem Pseudozufallsgenerator erzeugt, der mit einem **Seed** (Startwert) initialisiert wird. Derselbe Seed erzeugt deterministisch dieselben Fülltexte, dieselben Zufallswerte an denselben Positionen — und damit bit-identische Prompts. Das ist die Grundlage des fairen Vergleichs: Lauf A (BF16) und Lauf B (FP8) mit Seed 42 bearbeiten exakt dieselbe Aufgabe.

Ein einzelner Seed ist aber nur **eine** Stichprobe: Vielleicht liegen die Fakten bei Seed 42 zufällig günstig, bei Seed 45 ungünstig (z. B. sehr früh im Dokument, wo Degradation stärker zuschlägt). Deshalb unterstützt das Script einen Seed-Bereich (`SEED_MIN`/`SEED_MAX`): Jeder Seed erzeugt eine neue, unabhängige Aufgabenvariante mit anderen Werten und Positionen. Zehn Seeds ergeben pro Testkategorie zehn unabhängige Messpunkte — erst damit werden Aussagen wie "FP8 besteht T2 in 10/10 Fällen, BF16 ebenfalls" statistisch belastbar.

Davon zu unterscheiden ist `REPEATS`: Wiederholungen **desselben** Seeds messen nicht die Aufgabenvielfalt, sondern die Stabilität des Serving-Stacks. Trotz `temperature=0` kann Inferenz durch Batching-Effekte leicht nichtdeterministisch sein; wenn dieselbe Aufgabe mal besteht und mal nicht, ist das ein eigenes, relevantes Signal (Varianz), das man von echten Qualitätsunterschieden trennen muss. Faustregel: **Seeds beantworten die Qualitätsfrage, Repeats die Stabilitätsfrage.**

Alle Requests laufen mit `temperature=0`, um Sampling-Zufall zu minimieren, und mit identischem `max_tokens`, damit kein Lauf mehr "Denkbudget" bekommt als der andere.

## 4. Der A/B-Vergleich: So entsteht Beweiskraft

Beweiskraft entsteht durch Konstanthalten aller Variablen bis auf eine:

1. **Lauf A** gegen die Instanz mit FP8-KV fahren (z. B. `--kv-cache-dtype fp8_e4m3`).
2. **Lauf B** gegen dieselbe Instanz mit BF16-KV (`--kv-cache-dtype auto`) — gleiches Modell, gleiche Hardware, gleiche Serving-Version.
3. Beide Läufe mit **identischen** Werten für `SEED_MIN`, `SEED_MAX`, `CONTEXT_TOKENS`, `MAX_TOKENS` und `REPEATS`.
4. `CONTEXT_TOKENS` muss unter das kleinste beteiligte Kontextfenster passen, sonst vergleicht man unterschiedlich schwere Aufgaben — bei kürzerem Kontext liegen die Fakten dichter beieinander und die Tests werden leichter.
5. Zusätzlich lohnt ein Lauf mit sehr großem Kontext, den nur die FP8-Konfiguration erreichen kann: Dort würde akkumulierendes Quantisierungsrauschen zuerst sichtbar.

M�gliche Ergebnisse und ihre Bedeutung: Bestehen beide Konfigurationen gleich gut über alle Seeds und Kontextlängen, ist belegt, dass die KV-Kompression die gemessenen Fähigkeiten nicht verschlechtert — ein subjektiver Qualitätseindruck hätte dann eine andere Ursache. Fällt FP8 systematisch bei bestimmten Tests oder erst ab großen Kontextlängen ab, ist die Degradation real, lokalisiert und quantifiziert — dann sind gezielte Gegenmaßnahmen möglich (z. B. `fp8_e5m2`, kalibrierte KV-Skalierungsfaktoren oder ein kleineres Kontextfenster mit BF16). In beiden Fällen ersetzt der Test die Diskussion über Eindrücke durch Daten.

## 5. Bedienung

Alle Parameter werden als Umgebungsvariablen übergeben:

| Variable | Default | Bedeutung |
|---|---|---|
| `ENDPOINT` | `http://localhost:30000/v1/chat/completions` | OpenAI-kompatibler Chat-Completions-Endpoint (HTTP oder HTTPS) |
| `API_KEY` | *(leer)* | Optionaler Bearer-Token; wird als `Authorization: Bearer <key>` gesendet |
| `INSECURE` | `0` | `1` = `curl -k` für selbstsignierte/interne Zertifikate |
| `MODEL` | `glm-5.2` | Modellname, wie ihn der Endpoint/das Gateway erwartet |
| `CONTEXT_TOKENS` | `150000` | Ziel-Kontextlänge der generierten Dokumente. Achtung: Die interne Heuristik unterschätzt reale Tokenizer-Zählungen; mit dem GLM-Tokenizer wurden ~50 % mehr gemessen (`150000` → 223.147 `prompt_tokens`). Faustregeln: 131k-Fenster → max. ~85000, 190k-Fenster → max. ~120000. Verbindlich ist immer das `usage`-Feld der Server-Antwort (Spalte `prompt_tokens`) |
| `SEED` / `SEED_MIN` / `SEED_MAX` | `42` | Seed-Bereich (inklusiv); ohne Angabe läuft genau ein Seed |
| `REPEATS` | `1` | Wiederholungen pro Test und Seed (Stabilitätsmessung) |
| `MAX_TOKENS` | `4096` | Antwort-Budget pro Request (inkl. Thinking, je nach Server) |
| `CSV_FILE` | `kv_ab_results_<Zeitstempel>.csv` | Ergebnisdatei; existiert sie schon, wird angehängt |

Beispiel — Qualitätsvergleich mit 10 Seeds:

```bash
ENDPOINT=https://gateway.example.com/v1/chat/completions \
API_KEY=$KEY MODEL=glm-5.2 \
CONTEXT_TOKENS=120000 SEED_MIN=42 SEED_MAX=51 \
CSV_FILE=vergleich_120k.csv ./kv_ab_test.sh
```

Denselben Aufruf anschließend gegen die jeweils andere Konfiguration wiederholen — bei gleicher `CSV_FILE` landen alle Läufe in einer Datei und lassen sich nach `model`/`endpoint` gruppieren.

## 6. Ausgabe und Auswertung

Auf der Konsole erscheint pro Request eine Zeile mit Seed, Wiederholung, Testname, PASS/FAIL, erwartetem Wert, Latenz, Länge des Reasoning-Outputs in Zeichen, den serverseitig gezählten Prompt-Tokens sowie `finish=<finish_reason>/<completion_tokens>`. Bei FAIL wird zusätzlich der Anfang der Modellantwort gezeigt. Am Ende folgt eine Gesamtbilanz und eine Aufschlüsselung pro Testkategorie.

Die CSV enthält pro Request eine Zeile mit den Spalten `timestamp, endpoint, model, context_tokens, seed, repeat, test, verdict, expected, latency_s, reasoning_chars, finish_reason, completion_tokens, prompt_tokens, total_tokens, answer_snippet`. Für die Auswertung relevant:

Die **Pass-Rate pro Test und Konfiguration** ist die Hauptmetrik des A/B-Vergleichs. `reasoning_chars` zeigt, ob und wie ausführlich das Modell denkt — systematisch längeres Reasoning bei gleicher Aufgabe kann auf erschwerte Verarbeitung hindeuten, `0` bedeutet fehlenden Thinking-Output. `finish_reason` entlarvt technische Fehlerbilder: `length` bei leerer Antwort heißt, das Token-Budget wurde vollständig vom Denkprozess aufgebraucht, bevor eine Antwort entstand — ein Konfigurationsproblem, kein Intelligenzproblem. `prompt_tokens` ist die verbindliche Kontextlängen-Angabe für Vergleiche (Gateway-Dashboards zählen oft mit anderen Tokenizern und weichen ab). `latency_s` erlaubt nebenbei den Performancevergleich der Endpoints.

## 7. Grenzen der Aussagekraft

Der Test misst, was er misst: Retrieval-plus-Reasoning über synthetische Dokumente mit eindeutigen Antworten. Er misst nicht Schreibstil, Kreativität, Instruktionstreue bei komplexen Formatvorgaben oder domänenspezifisches Wissen. Ein vollständig bestandener Lauf beweist, dass die Kernmechanik des Schlussfolgerns über lange Kontexte in der getesteten Konfiguration intakt ist — er beweist nicht, dass jeder erdenkliche Prompt in beiden Konfigurationen identische Ergebnisse liefert. Für ein abschließendes Urteil gehören reale Prompts aus dem tatsächlichen Workload als zusätzliche Testfälle in den Vergleich.


