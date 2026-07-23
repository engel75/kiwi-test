#!/usr/bin/env bash
# =============================================================================
# KV-Cache A/B-Test: BF16 vs FP8-e4m3 — Reasoning/Aggregation im Langkontext
#
# Testet gezielt die Faehigkeiten, die NIAH NICHT abdeckt:
#   T1  Variable-Tracking   (Zustand ueber viele Reassignments verfolgen)
#   T2  Aggregation         (verstreute Zahlen aufsummieren)
#   T3  Multi-Hop           (3-stufige Schlusskette ueber verteilte Fakten)
#   T4  Reasoning-Sanity    (Kurzkontext-Logik als Baseline)
#   T5  Thinking-Check      (liefert der Reasoning-Parser Reasoning-Output?)
#
# Nutzung:
#   ENDPOINT=https://ai.ewcs.ch/v1/chat/completions \
#   API_KEY=sk-... MODEL=ew/glm-5.2 \
#   CONTEXT_TOKENS=120000 SEED_MIN=42 SEED_MAX=47 REPEATS=10 ./kiwi-test.sh
#
#   CONTEXT_TOKENS  Ziel-Prompt-Laenge; Generator ist auf den GLM-Tokenizer
#               kalibriert (TOKENS_PER_SENT=21, gemessen). Andere Tokenizer:
#               TOKENS_PER_SENT anpassen. ACHTUNG: Aenderung der Kalibrierung
#               aendert die generierten Prompts pro Seed — innerhalb eines
#               A/B-Vergleichs konstant halten!
#   API_KEY     optional — wird als "Authorization: Bearer <key>" gesendet
#   INSECURE=1  optional — curl -k fuer selbstsignierte/interne Zertifikate
#   SEED_MIN/SEED_MAX  Seed-Bereich (inklusiv); Default: SEED (=42)
#   REPEATS     Wiederholungen pro Test+Seed (Default 1) — misst Varianz/
#               Nichtdeterminismus des Serving-Stacks trotz temperature=0
#   CSV_FILE    Pfad der CSV-Ausgabe (Default: kiwi_results_<ts>.csv im CWD)
#
# Gleiche Aufrufe (identische Seeds!) gegen BF16 und FP8 fahren, CSVs vergleichen.
# =============================================================================
set -uo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:30000/v1/chat/completions}"
API_KEY="${API_KEY:-}"        # optional: Bearer-Token
INSECURE="${INSECURE:-0}"     # 1 = curl -k (selbstsignierte Zertifikate)
MODEL="${MODEL:-glm-5.2}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-150000}"   # Ziel-Kontextlaenge der Tests
TOKENS_PER_SENT="${TOKENS_PER_SENT:-21}"     # kalibriert auf GLM-Tokenizer (gemessen)
SEED="${SEED:-42}"
SEED_MIN="${SEED_MIN:-$SEED}"
SEED_MAX="${SEED_MAX:-$SEED_MIN}"
REPEATS="${REPEATS:-1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
CSV_FILE="${CSV_FILE:-kiwi_results_$(date +%Y%m%d_%H%M%S).csv}"
TMPDIR_AB="$(mktemp -d /tmp/kiwi_test.XXXXXX)"
trap 'rm -rf "$TMPDIR_AB"' EXIT

PASS=0; FAIL=0; TRUNC=0
declare -A TPASS TFAIL TTRUNC

# --- CSV -------------------------------------------------------------------
csv_escape() { local s="${1//\"/\"\"}"; printf '"%s"' "$s"; }
if [[ ! -f "$CSV_FILE" ]]; then
  echo "timestamp,endpoint,model,context_tokens,seed,repeat,test,verdict,expected,latency_s,reasoning_chars,finish_reason,completion_tokens,prompt_tokens,total_tokens,answer_snippet" > "$CSV_FILE"
fi

# --- Payload-Generator (Python fuer sauberes JSON-Escaping + Fuelltext) ------
gen_payload() {  # $1=testname $2=seed  -> schreibt payload.json + expected.txt
  python3 - "$1" "$CONTEXT_TOKENS" "$2" "$TMPDIR_AB" "$MODEL" "$MAX_TOKENS" "$TOKENS_PER_SENT" <<'PYEOF'
import json, random, sys
test, ctx_tokens, seed, tmpdir, model, max_tokens, tok_per_sent = sys.argv[1:8]
ctx_tokens, seed, max_tokens = int(ctx_tokens), int(seed), int(max_tokens)
tok_per_sent = int(tok_per_sent)
rng = random.Random(seed)

FILLER = [
    "Die Wartung der Kuehlsysteme erfolgt quartalsweise durch das Facility-Team.",
    "Das Change-Advisory-Board tagt jeden zweiten Donnerstag im Monat.",
    "Alle Deployments in der Produktionsumgebung erfordern ein Vier-Augen-Prinzip.",
    "Die Backup-Rotation folgt einem Grossvater-Vater-Sohn-Schema mit Offsite-Kopie.",
    "Netzwerksegmente werden gemaess der internen Zonenrichtlinie getrennt betrieben.",
    "Der Bereitschaftsdienst rotiert woechentlich zwischen den Teammitgliedern.",
    "Zertifikate werden neunzig Tage vor Ablauf automatisch zur Erneuerung gemeldet.",
    "Die Kapazitaetsplanung wird halbjaehrlich mit den Fachabteilungen abgestimmt.",
]

def filler_block(n_sent):
    return " ".join(rng.choice(FILLER) for _ in range(n_sent))

# kalibriert: tok_per_sent Tokens pro Fuellsatz -> Saetze gesamt
total_sents = max(50, ctx_tokens // tok_per_sent)
n_facts_slots = 40
sents_per_chunk = max(1, total_sents // (n_facts_slots + 1))

def build_doc(facts):
    """facts: Liste von Strings, gleichmaessig ueber das Dokument verteilt."""
    parts = []
    slots = sorted(rng.sample(range(n_facts_slots), len(facts)))
    fi = 0
    for slot in range(n_facts_slots):
        parts.append(filler_block(sents_per_chunk))
        if fi < len(facts) and slot == slots[fi]:
            parts.append(facts[fi]); fi += 1
    parts.append(filler_block(sents_per_chunk))
    return "\n\n".join(parts)

if test == "t1_variable_tracking":
    vals = [rng.randint(100, 999) for _ in range(12)]
    facts = [f"[SYSTEMLOG] Parameter ALPHA wurde auf den Wert {v} gesetzt." for v in vals]
    doc = build_doc(facts)  # Reihenfolge im Doc = Reihenfolge der facts-Liste
    q = ("Im Dokument wird der Parameter ALPHA mehrfach neu gesetzt. "
         "Welchen Wert hat ALPHA am Ende des Dokuments (letzte Zuweisung)? "
         "Antworte nur mit der Zahl.")
    expected = str(vals[-1])

elif test == "t2_aggregation":
    amounts = [rng.randint(10, 500) for _ in range(15)]
    facts = [f"[BUCHUNG] Rechnung Nr. {1000+i}: Betrag {a} CHF." for i, a in enumerate(amounts)]
    doc = build_doc(facts)
    q = ("Summiere die Betraege ALLER [BUCHUNG]-Eintraege im Dokument. "
         "Antworte nur mit der Gesamtsumme in CHF als Zahl.")
    expected = str(sum(amounts))

elif test == "t3_multihop":
    code = rng.randint(10000, 99999)
    facts = [
        "[NOTIZ] Der Serverschluessel wird von Mitarbeiterin Verena Kolb verwahrt.",
        "[NOTIZ] Verena Kolb arbeitet am Standort Rapperswil.",
        f"[NOTIZ] Der Tresorcode am Standort Rapperswil lautet {code}.",
    ]
    rng.shuffle(facts)
    doc = build_doc(facts)
    q = ("Welcher Tresorcode ist noetig, um an den Serverschluessel zu gelangen? "
         "Leite die Antwort aus den [NOTIZ]-Eintraegen ab und antworte nur mit der Zahl.")
    expected = str(code)

elif test == "t4_sanity":
    doc = ""
    q = ("Anna ist doppelt so alt wie Ben. In 6 Jahren ist Anna nur noch "
         "anderthalb mal so alt wie Ben. Wie alt ist Anna heute? "
         "Antworte am Ende nur mit der Zahl.")
    expected = "12"

elif test == "t5_thinking":
    doc = ""
    q = "Was ist 17 * 23? Denke Schritt fuer Schritt."
    expected = "391"
else:
    sys.exit(f"Unbekannter Test: {test}")

content = (f"Hier ist ein internes Dokument:\n\n{doc}\n\n---\nFrage: {q}"
           if doc else q)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "temperature": 0,
    "max_tokens": max_tokens,
}
with open(f"{tmpdir}/payload.json", "w") as f:
    json.dump(payload, f)
with open(f"{tmpdir}/expected.txt", "w") as f:
    f.write(expected)
PYEOF
}

run_test() {  # $1=testname $2=seed $3=repeat  (payload.json muss bereits existieren)
  local name="$1" seed="$2" rep="$3"
  local expected; expected="$(cat "$TMPDIR_AB/expected.txt")"

  local -a curl_args=(-sS --max-time 600 -H 'Content-Type: application/json')
  [[ -n "$API_KEY" ]] && curl_args+=(-H "Authorization: Bearer $API_KEY")
  [[ "$INSECURE" == "1" ]] && curl_args+=(-k)

  local t0 t1 resp
  t0=$(date +%s.%N)
  resp="$(curl "${curl_args[@]}" "$ENDPOINT" \
    -d @"$TMPDIR_AB/payload.json")" || { echo "CURL-FEHLER: $name (seed=$seed rep=$rep)"; FAIL=$((FAIL+1)); TFAIL[$name]=$(( ${TFAIL[$name]:-0} + 1 )); return; }
  t1=$(date +%s.%N)
  local latency; latency="$(echo "$t1 - $t0" | bc)"

  local answer reasoning_len fr ct pt tt
  answer="$(echo "$resp" | python3 -c '
import json,sys
d=json.load(sys.stdin)
m=d["choices"][0]["message"]
print((m.get("content") or "").strip())' 2>/dev/null)"
  reasoning_len="$(echo "$resp" | python3 -c '
import json,sys
d=json.load(sys.stdin)
m=d["choices"][0]["message"]
r = m.get("reasoning_content") or m.get("reasoning") or ""
if not r:
    r = "".join(x.get("text","") for x in (m.get("reasoning_details") or []))
print(len(r))' 2>/dev/null)"
  read -r fr ct pt tt <<< "$(echo "$resp" | python3 -c '
import json,sys
d=json.load(sys.stdin)
c=d["choices"][0]
u=d.get("usage") or {}
print(str(c.get("finish_reason")), str(u.get("completion_tokens","?")), str(u.get("prompt_tokens","?")), str(u.get("total_tokens","?")))' 2>/dev/null)"

  local verdict="FAIL"
  if [[ "$name" == "t5_thinking" ]]; then
    # Bestanden, wenn Antwort korrekt UND Reasoning-Output vorhanden
    if echo "$answer" | grep -q "$expected" && [[ "${reasoning_len:-0}" -gt 0 ]]; then verdict="PASS"; fi
  else
    echo "$answer" | grep -q "$expected" && verdict="PASS"
  fi
  # Budget-Abbruch von echtem Reasoning-Fehler trennen: finish_reason=length ohne
  # korrekte Antwort heisst "beim Denken abgeschnitten", nicht "falsch gedacht".
  # TRUNC-Faelle mit hoeherem MAX_TOKENS wiederholen statt als Fehler zu werten.
  if [[ "$verdict" == "FAIL" && "${fr:-}" == "length" ]]; then verdict="TRUNC"; fi
  case "$verdict" in
    PASS)  PASS=$((PASS+1));   TPASS[$name]=$(( ${TPASS[$name]:-0} + 1 ));;
    TRUNC) TRUNC=$((TRUNC+1)); TTRUNC[$name]=$(( ${TTRUNC[$name]:-0} + 1 ));;
    *)     FAIL=$((FAIL+1));   TFAIL[$name]=$(( ${TFAIL[$name]:-0} + 1 ));;
  esac

  printf 'seed=%-4s rep=%-3s %-22s %-4s  erwartet=%-8s  latenz=%5.1fs  reasoning_chars=%-6s prompt=%stok finish=%s/%stok\n' \
    "$seed" "$rep" "$name" "$verdict" "$expected" "$latency" "${reasoning_len:-?}" "${pt:-?}" "${fr:-?}" "${ct:-?}"
  [[ "$verdict" != "PASS" ]] && printf '  -> Antwort (gekuerzt): %.300s\n' "$answer"

  local snip; snip="$(echo "$answer" | tr '\n\r' '  ' | cut -c1-120)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$(csv_escape "$ENDPOINT")" "$(csv_escape "$MODEL")" \
    "$CONTEXT_TOKENS" "$seed" "$rep" "$name" "$verdict" "$(csv_escape "$expected")" \
    "$latency" "${reasoning_len:-}" "$(csv_escape "${fr:-}")" "${ct:-}" "${pt:-}" "${tt:-}" \
    "$(csv_escape "$snip")" >> "$CSV_FILE"
}

TESTS=(t1_variable_tracking t2_aggregation t3_multihop t4_sanity t5_thinking)
TOTAL=$(( (SEED_MAX - SEED_MIN + 1) * REPEATS * ${#TESTS[@]} ))
echo "Endpoint: $ENDPOINT | Modell: $MODEL | Kontext: ~${CONTEXT_TOKENS} Tokens"
echo "Seeds: $SEED_MIN..$SEED_MAX | Wiederholungen: $REPEATS | Gesamt: $TOTAL Requests"
echo "CSV: $CSV_FILE"
echo "-----------------------------------------------------------------------"

for seed in $(seq "$SEED_MIN" "$SEED_MAX"); do
  for name in "${TESTS[@]}"; do
    gen_payload "$name" "$seed" || { echo "GEN-FEHLER: $name (seed=$seed)"; FAIL=$((FAIL+1)); continue; }
    for rep in $(seq 1 "$REPEATS"); do
      run_test "$name" "$seed" "$rep"
    done
  done
done

echo "-----------------------------------------------------------------------"
echo "Ergebnis gesamt: $PASS PASS / $FAIL FAIL / $TRUNC TRUNC (Budget-Abbruch)  (von $TOTAL)"
for name in "${TESTS[@]}"; do
  printf '  %-22s %s PASS / %s FAIL / %s TRUNC\n' "$name" "${TPASS[$name]:-0}" "${TFAIL[$name]:-0}" "${TTRUNC[$name]:-0}"
done
echo "CSV geschrieben: $CSV_FILE"
[[ "$TRUNC" -gt 0 ]] && echo "Hinweis: TRUNC = Thinking hat MAX_TOKENS ($MAX_TOKENS) aufgebraucht, bevor eine Antwort entstand. Diese Faelle mit hoeherem MAX_TOKENS wiederholen." || true

#EOF
