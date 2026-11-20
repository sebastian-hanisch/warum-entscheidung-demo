# Warum ist dieses Paket drin? Eine Packentscheidung begründen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-warum-entscheidung-demo.streamlit.app/)** -->

Stück der Linie **„Beitrag und Beleg“** (Feld „Modelle verstehen“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Ast „von Natur erklärbar und Entscheidungen“. Die Demo begründet eine **Packentscheidung** (Rucksack „Fuhre packen“) **exakt** und zählt, **wo die Begründung bricht**: bei Gleichstand, bei mehr Kapazität, bei einer einfachen Regel und bei nur geschätzten Werten. Kante der Linie: `fix` auf „Was müsste anders sein?“ (dort ist der kleinste Eingriff, der eine **Modell**entscheidung kippt, am brüchigsten, weil das Modell sich beim Neu-Training ändert; hier gibt es **kein Modell**, die Annahme lautet: *wenn die Daten stimmen, hat jede Entscheidung genau eine Begründung, und sie gilt auch bei kleiner Änderung*) und `kontrast` zu „Dünner Baum gegen Boosting“ (dort ist das Modell selbst lesbar, hier ist die **Regel** das lesbare Modell der Entscheidung).

**Es gibt keine reale Datenquelle und kein Modell.** Die Pakete sind ausgedacht: Gewichte und Werte sind ganze Zahlen aus einem festen Zufallsstrom (SplitMix64), und die richtige Antwort jeder Frage wird durch Aufzählung aller Mengen oder durch dynamische Programmierung berechnet, nicht geschätzt. Deshalb lässt sich zeigen, wann eine Begründung der Entscheidung stimmt und wann sie bricht. In den gelesenen Quellen (nur Zusammenfassungen) fand sich kein Beleg dafür, dass solche Begründungen in Betrieb oder Logistik bessere Entscheidungen liefern; die Demo behauptet das nicht, und alle Aussagen gelten nur für diese Pakete. **Keine Neuheitsaussage:** Sensitivitätsanalyse und Gegenfaktum sind Lehrbuch und Forschung; neu ist hier nur die nachrechenbare Vorführung mit Zähleinheit und Orakeln.

## Warum dieses Problem

Eine Optimierungsentscheidung wirkt eindeutig: Paket A ist drin, Paket B nicht. Die Frage „Warum?“ hat aber nur mit einer **Lesart** eine Antwort: **sicher drin** (in jeder optimalen Menge), **sicher draußen** (in keiner) oder **Gleichstand** (in mancher). **Kernfrage:** Wie oft ist die Begründung eindeutig, und wie stabil ist sie, wenn die Kapazität wächst, wenn man eine einfache Regel statt der exakten Lösung benutzt oder wenn die Werte nur geschätzt sind?

## Modell

* **Vehikel „Fuhre packen“ (erfunden):** n Pakete mit ganzzahligem Gewicht w in [5s, 30s] und Wert v in [10s, 60s] (**unkorreliert**) oder v = w + 10s (**stark korreliert**); Kapazität C = ⌊Anteil · Σw⌋ mit Anteil 0,3 / 0,5 / 0,7; s ist die **Zahlenskala** (1 grob, 10 fein). Alles ganzzahlig, deshalb sind Gleichstände zählbar. Die Familien sind **nicht wortgleich** mit denen anderer Demos der Rucksack-Linie (`branch-bound-demo`, `fptas-rucksack-demo`).
* **Status** eines Pakets bei Kapazität C: +1 sicher drin, −1 sicher draußen, 0 Gleichstand. **Wertschwelle** (Kippabstand) d = |A − B| mit A = bester Wert mit Paket i und B = bester Wert ohne Paket i (beide bei derselben Kapazität). Ändert man den Wert von Paket i um d − 1, bleibt der Status; bei d entsteht Gleichstand; bei d + 1 ist er umgekehrt (Test). Für ein sicher draußen liegendes Paket ist der **Preis der Erzwingung** („Paket i zwingend packen“) B − A und damit **dieselbe Zahl** (Identität, Test); bei einem sicher drin liegenden Paket kostet das Erzwingen 0, bei Gleichstand ebenfalls 0.
* **Wahrheit je Funktional** heißt hier: die **exakte Lösung der Instanz** mit festgelegter Lesart; sie hat keinen Modellfehler (Aufzählung aller 2^n Mengen bis n = 12, dynamische Programmierung bis n = 40, ein ganzzahliges Programm im Test). Ein „Beitrag“ oder eine „Wichtigkeit“ der Linie kommt nicht vor (kein Modell, kein Shapley-Wert); ersatzweise zeigt die App den Kippabstand **absolut** (Wertpunkte) und **relativ** (Anteil des eigenen Werts).
* **Regeln:** Greedy nach Verhältnis v/w (füllt Lücken) und Regel S (bis zum kritischen Paket). **Reihenfolge (Tab):** Einmaschinen-Problem 1 | r_j | Σ w_j C_j mit n = 7 Aufträgen (5 040 Reihenfolgen); statische Smith-Reihenfolge und Dispatch.
* **Schätzung:** v̂ = rint(v · (1 + δ·U(−1,1))); Gewichte und Kapazität bekannt; ein Rauschmodell, kein Lernmodell. Aktion: den wahren Wert um m_hat · (1 + Aufschlag) beziehungsweise m_hat + κ·δ·(mittlerer geschätzter Wert) anheben; gültig, wenn die Anhebung knapp über m_aktion liegt und m_aktion ≥ m_wahr ist.
* **Zähleinheiten:** Anteile über **Instanzen** (je Ziehung 3 Kapazitätsanteile × 100 = 300; bei n = 40: 150) oder **Pakete** (Instanz × Paket mit eindeutigem Status), **Neulösungen** (Instanz × 10 Störungen), **Störungen** (Instanz × Paket × Schätzung, 5 Schätzungen je Instanz). Jede Zelle (n, Zahlenskala, Familie) wird in **fünf getrennten Ziehungen** gemessen; angegeben ist das **Mittel [kleinste bis größte Ziehung]**, nie die Punktzahl einer festen Ziehung. Keine Sekunden als Befund.

## Methodik

Die Einzelinstanz rechnet live (Aufzählung bis 12 Pakete, dynamische Programmierung bis 40); die Ensemble-Zahlen stehen im **vorgerechneten Raster** `data/wde_vorgerechnet.json` (Skript `tools/wde_make_vorgerechnet.py`, 175 Aufgaben, einige Minuten; alle Zähler sind Ganzzahlen aus SplitMix64 und exakter Arithmetik und auf jeder Plattform gleich). Fünf Experimente: **(1)** Warum ist Paket A drin, B nicht? (Tabelle mit Status, Kippabstand, Preis der Erzwingung; Handbeispiel mit vier Paketen); **(2)** Ist die Antwort eindeutig? (Gleichstand); **(3)** Was ändert mehr Kapazität? (Statusstreifen über alle Kapazitäten); **(4)** Reicht eine einfache Regel? (Tab „Reihenfolge“); **(5)** Gilt die Begründung auf Schätzwerten?

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Mittel [kleinste bis größte der fünf Ziehungen] aus dem Raster; „unk“ = unkorreliert, „stark“ = stark korreliert, s = Zahlenskala. Die Spalte „Test“ nennt den Test in `tests/test_claims.py`. Die Zahlen gelten für **dieses erfundene Vehikel, n = 8 bis 40, die genannten Skalen und Familien**; über Betrieb, Logistik oder Recht sagen sie nichts.

| Frage | Befund | Test |
|---|---|---|
| Stimmen Status und Wertschwelle? | Dynamische Programmierung = Aufzählung (Python-Schleife) in 48 Instanzen (n = 6, 8, beide Familien und Skalen, drei Kapazitäten); Wertschwelle d − 1 / d / d + 1 gegen Aufzählung in 62 von 62 Paketen; Preis der Erzwingung = Wertschwelle bei sicher draußen liegenden Paketen (bei sicher drin und Gleichstand 0) in 64 von 64 Paketen; Raster neu gerechnet: exakt gleiche Zähler | `tests/test_oracle_rucksack.py`, `test_neu_rechnen_gibt_exakt_dieselben_zaehler` |
| Hat jede Instanz eine eindeutige optimale Menge? (Experiment 2) | **Nein.** Instanzen mit mehreren optimalen Mengen, unk, s = 1, n = 8 / 12 / 20 / 40: 4,3 % [2,3 bis 6,0] / 6,4 % [4,3 bis 9,0] / 10,8 % [9,3 bis 13,0] / 19,9 % [13,3 bis 28,7]; mit s = 10: 0,5 / 0,8 / 1,4 / 1,9 %. **Stark, s = 1: 41,7 / 74,5 / 88,8 / 96,0 %**, s = 10: 6,4 / 21,2 / 56,7 / 86,1 % | `test_gleichstand` |
| Gleichstand durch identische Pakete | Ein Teil der Gleichstände ist nur das Vertauschen identischer Pakete (gleiches Gewicht und gleicher Wert): bei stark korrelierten Zahlen und s = 1 beruht ein erheblicher Teil darauf (n = 8: 41,7 % gegen 20,1 % ohne Vertauschung; Nachzählung der unabhängigen Prüfung, nicht im Raster und nicht im Test) | keiner |
| Mischwert | Die vier Zellen n = 8, 12, s = 1, beide Familien zusammen: 31,7 % [30,6 bis 33,2] der Instanzen; als einzelne Zahl nicht lesbar (die Zellen gehen von 4,3 bis 74,5 % auseinander) | `test_gleichstand` |
| Kippabstand (Experiment 1) | Median des relativen Kippabstands (Pakete mit eindeutigem Status), unk: 0,47 bis 0,57 des eigenen Werts über alle acht Zellen; Anteil mit Abstand höchstens 5 %: 3,2 bis 5,4 %. stark: 0,04 bis 0,43 beziehungsweise 1,7 bis 53,2 % | `test_kippabstand` |
| Mehr Kapazität (Experiment 3) | Pakete, die irgendwo im Bereich 0 bis Σw bei mehr Kapazität hinausfliegen, unk, s = 1: 73,9 / 83,6 / 91,3 / 96,8 % (n = 8 bis 40; der Bereich wächst mit n); **im Nahbereich** (Status ändert sich schon bei höchstens 5 % Kapazitätsänderung): unk 20,7 bis 29,1 % der Pakete, stark 69,2 bis 99,4 % | `test_kapazitaet` |
| Einfache Regel (Experiment 4), unk, s = 1, n = 8 / 12 / 20 / 40 | Greedy: Pakete **richtig** 91,0 / 91,1 / 93,6 / 96,1 %, **ganze Menge gleich** (Instanzen mit eindeutigem Optimum) 68,3 / 55,3 / 46,4 / 32,6 %, Optimalwert getroffen 68,4 / 55,1 / 46,1 / 33,2 %; Basis „Pakete richtig“ (das häufigere Status raten) 56,6 bis 60,2 % | `test_regeln` |
| … stark korreliert | Greedy trifft den Optimalwert nur in 2,4 bis 12,7 % der Instanzen (alle acht Zellen), Pakete richtig 72,2 bis 99,8 %; bei n = 40, s = 1 gibt es nur 5 bis 8 Instanzen mit eindeutigem Optimum je Ziehung: „ganze Menge gleich“ ist dort nicht belastbar | `test_regeln` |
| Mischwert Greedy | Greedy trifft den Optimalwert über die vier Zellen n = 8, 12, s = 1, beide Familien: 37,2 % [35,7 bis 39,8] | `test_regeln` |
| Reihenfolge (Tab), n = 7, 200 Instanzen je Ziehung | Ohne Freigaben: statische Smith-Reihenfolge und Dispatch **exakt** (100 % der Instanzen und aller sicheren Paare); mehrere optimale Reihenfolgen trotzdem in 58,8 % [56,0 bis 61,0] (gezählt ist auch das Vertauschen identischer Aufträge). Freigaben bis 10: statisch trifft den Optimalwert in 8,6 % [7,5 bis 9,0], Dispatch in 61,1 % [59,5 bis 63,0]; bis 25: 1,2 % [0,5 bis 2,0] gegen 54,0 % [51,0 bis 60,5] | `test_reihenfolge` |
| Gegenfaktum der Reihenfolge (nur ohne Freigaben, geschlossene Formel) | Kleinste ganzzahlige Senkung von w_A, relativ, im Median 0,75; Anteil höchstens 1/3: 14,3 %; kleinste Erhöhung von p_A relativ im Median 1,5 (Mittel der fünf Ziehungsmediane 1,45); 645 von 20 090 sicheren Paaren (fünf Ziehungen) nicht kippbar bis +29 | `test_reihenfolge` |
| Störung aller Werte (Neulösungen) | Optimale Menge ändert sich bei Störung ±5 % / ±10 %, unk, s = 1, n = 12 / 20 / 40: 5,9 / 9,9 / 18,6 % und 14,7 / 22,8 / 38,7 %; stark, s = 10, schon bei ±1 %: 26,0 / 28,9 / 31,5 % | `test_stoerung` |
| Schätzwerte (Experiment 5), Aufschlag 0 % | Die auf Schätzwerten berechnete Anhebung gilt an den wahren Werten bei unk in 51,2 bis 63,1 % der Störungen (acht Zellen), bei stark mit n = 12 in 59,5 bis 68,8 %, **bei stark mit n = 40 nur in 20,2 bis 35,1 %** | `test_schaetzung` |
| … Verdopplung (relativer Aufschlag 100 %) | unk 95,2 bis 99,2 %; **stark mit n = 40: 69,0 bis 90,2 %** (selbst bei Verdopplung nicht verlässlich; unter 95 % liegt sie in 6 von 16 Zellen, alle stark korreliert); Aufwand bis zum 2,6-Fachen | `test_schaetzung` |
| … absoluter Abstand | **8 Rauscheinheiten: in allen 16 Zellen kein einziger Fehlschlag** (14 569 bis 118 187 Störungen je Zelle aus fünf Ziehungen); der Aufwand reicht bis zum 12,1-Fachen. Das heißt nicht „sicher“, sondern „in dieser Messung nie ein Fehlschlag“, und es setzt ein bekanntes Rauschniveau voraus | `test_schaetzung` |

**Handbeispiel (live in der App):** Gewichte 4, 3, 3, 5, Werte 6, 7, 9, 9, Kapazität 8: Optimum 18 mit {3, 4}, eindeutig; Paket 2 (Wert 7) liegt in keiner optimalen Menge, seine Wertschwelle ist 2 (B = 18, A = 7 + 9 = 16); bei Wert 8 bleibt es draußen, bei 9 **Gleichstand mit drei optimalen Mengen** {3, 4}, {2, 3}, {2, 4} (je 18), bei 10 in jeder optimalen Menge; das Paket zu erzwingen kostet genau 2. Greedy nimmt {2, 3} mit Wert 16 gegen 18 und trifft nur zwei der vier Einzelentscheidungen. **Paket 2 ist bei Kapazität 6 und 7 drin, bei 8 und 9 draußen, bei 10 wieder drin:** der Optimalwert steigt von 16 auf 18, trotzdem fliegt es hinaus. Schätzung (6, 8, 9, 9): geschätzte Schwelle 1 gegen wahre 2; gültig ab Aufschlag 100 %. Test: `tests/test_oracle_handbeispiel.py` (gegen Handrechnung und eine zweite Aufzählung).

## Befunde und Korrekturen gegenüber dem Plan und der Vormessung

Plan und Vormessung liefern **Erwartungen**; neu gemessen wurde mit dem Zufall der Demo (SplitMix64) und den Zufalls-Seeds der Korrekturrunde (fünf Ziehungen).

* **Gleichstand und Greedy (unk, s = 1):** Die Bandtabellen des Plans (Korrekturrunde) sind **bis auf die letzte Stelle bestätigt** (gleiche Zufalls-Seeds, eigener Code); sie stehen jetzt aus dem Raster in der App. Der Mischwert 31,7 % [30,6 bis 33,2] stimmt, der zweite Mischwert (Greedy) liegt bei 37,2 % [35,7 bis 39,8] statt 37,1.
* **Kapazität, Kippabstand und Störung:** Der Plan zitierte hier **eine** Ziehung mit anderen Zufalls-Seeds (z. B. Nichtmonotonie unk, s = 1: 72,1 / 83,6 / 91,4 / 97,0 %; Median des Kippabstands 0,612 / 0,513 / 0,500 / 0,491). Der Bau misst dieselben Größen über fünf Ziehungen: 73,9 / 83,6 / 91,3 / 96,8 % und 0,57 / 0,51 / 0,50 / 0,49 (Mittel der Ziehungsmediane); die Einzelziehung des Plans liegt bei n = 8 außerhalb des Mittels der Ziehungen (0,612 gegen 0,571): ein Hinweis, die Ziehungen nicht einzeln zu zitieren.
* **Reihenfolge:** Plan (eine Ziehung, 200 Instanzen) 61,0 ± 3,6 / 34,0 ± 3,4 / 24,5 ± 3,2 % mehrere optimale Reihenfolgen bei Freigaben bis 0 / 10 / 25; Bau 58,8 [56,0 bis 61,0] / 35,7 [32,0 bis 40,5] / 21,5 [19,5 bis 24,5] %: die Zahl hängt an der Ziehung, die Größenordnung hält. Ziehung 0 des Baus benutzt die Zufalls-Seeds des Plans und reproduziert dessen Zahlen: 122 von 200 Instanzen ohne Freigaben mit mehreren optimalen Reihenfolgen (61,0 %), 4 012 sichere Paare, 152 davon nicht kippbar bis +29, Median der Senkung von w_A 0,75, Anteil höchstens 1/3: 15,1 %.
* **Schätzwerte:** Der Plan sagte „volle Sicherheit gibt es nicht“ nur für den relativen Aufschlag bei stark korrelierten Zahlen und n = 40 (69 bis 90 % bei Verdopplung): bestätigt (69,0 bis 90,2 %); ein Abstand von 8 Rauscheinheiten hält in allen 16 Zellen und fünf Ziehungen jede Anhebung. Der Aufwand reicht im Bau bis zum 12,1-Fachen (Plan: 11,7).
* **Offene Frage des Plans „Basisrate bei n ≥ 20“:** Die Basis (das häufigere Status raten, 56,6 bis 60,2 % bei unk, s = 1) liegt weit unter „Pakete richtig“ (91,0 bis 96,1 %): Greedy ist je Paket klar besser als Raten, auch bei n = 20 und 40.
* **Regel S gegen Greedy:** Greedy enthält die Menge der Regel S und hat nie einen kleineren Wert; trifft S den Optimalwert, trifft ihn auch Greedy (Test über alle 80 Zellen-Ziehungen). Bei den **Einzelentscheidungen** ist Regel S in 8 von 80 Zellen-Ziehungen minimal besser (Greedy füllt Lücken mit Paketen, die in der optimalen Menge fehlen); bei der ganzen Menge nie.
* **Nicht gebaut:** das Folgestück „Warum nicht? Kontext und Unzulässigkeit“ (nächster Kontext, irreduzible unzulässige Teilsysteme); Entscheidung E17 des Plans (Teilung) ist offen, der Bau hat **nicht geteilt**.

## Ehrliche Grenzen

* **Kein Modell, erfundene Zahlen:** nichts über Nutzen oder Gefahr solcher Begründungen in Betrieb, Logistik oder Recht; Lesbarkeit für Menschen ist nicht gemessen.
* **Ein Problem, ein Tab:** Rucksack mit n = 8 bis 40, Zahlenskalen 1 und 10, zwei Familien; Reihenfolge nur mit 7 Aufträgen und dem Ziel Σ w C. Größere Instanzen, Skalen über 10 und andere Ziele (Verspätung) sind nicht gemessen; das Raster zu Experiment 5 gibt es nur für n = 12 und 40.
* **Schätzwert-Falle:** nur das multiplikative gleichverteilte Rauschen mit bekanntem Niveau; Rauschen, das mit dem Wert wächst oder verzerrt ist, ist nicht gemessen. Der absolute Abstand setzt ein bekanntes δ voraus.
* **Zahl der Instanzen:** Je Ziehung 300 Instanzen (n = 40: 150); die Spanne über fünf Ziehungen ist breit, wo wenige Instanzen eindeutig sind (n = 40, stark, s = 1: 5 bis 8 je Ziehung).
* **Gleichstand zählt als Änderung** beim Kapazitäts-Nahbereich; der Bereich 0 bis Σw wächst mit n, deshalb ist die Nichtmonotonie-Zahl allein zwischen verschiedenen n nicht vergleichbar, der Nahbereich ist die bessere Größe.
* **LP-Begründungen** (Schattenpreise, Bereiche) sind nicht Teil dieser Demo; siehe `lp-dualitaet-demo`. Ihre Brüchigkeitszahlen sind mit denen dieser Demo nicht vergleichbar.

## Verifikation

**Orakel** (Python-Schleife über alle Mengen, Aufzählung aller 5 040 Reihenfolgen, `scipy.optimize.milp` nur im Test): Status, Wertschwelle bei d − 1 / d / d + 1, Preis der Erzwingung, schnelle Schwellen gegen die Statustabelle über alle Kapazitäten, Smith-Regel gegen Aufzählung, geschlossene Gegenfakten gegen Aufzählung, Big-M-Programm gegen Aufzählung, Handbeispiel in allen Teilen. **Eigenschaften** (Werte, Gewichte, Kapazität skaliert, Pakete umnummeriert, identische Pakete, mehr Kapazität senkt den Optimalwert nie und der Status ist nicht monoton, Kippabstand null genau im Gleichstand), **Randfälle** (C = 0, Paket über der Kapazität, n = 1, alle Werte gleich, Gleichstand zwischen allen Mengen), **Raster** (Invarianten der Zähler, Quelltext-Hash, exaktes Neu-Rechnen je einer Zelle jeder Art), **Claims** (jede Zahl dieser README), **Verdikte**, **Oberfläche** (AppTest: jede Option jedes Reglers, jedes Preset), **Layout**, Footer, Laufzeit-Importe.

## Tests

**179 Tests** (`pytest tests`, rund 90 Sekunden mit `-n 4`, gemessen unter wechselnder Last auf einem Windows-Rechner mit Python 3.14): Orakel 67 (Rucksack 53, Handbeispiel 6, Reihenfolge 8), Eigenschaften und Randfälle 17, Raster 11, Claims 15, Auswertung und Verdikte 19, Oberfläche, Presets, Layout, Footer und Laufzeit-Importe 50. Die Zahlen der README stehen in `tests/test_claims.py` als Messung des Rasters; das exakte Neu-Rechnen je einer Zelle jeder Art steht in `tests/test_grid.py` (alle Zähler gleich, auf jeder Plattform). Das ganzzahlige Programm (`scipy.optimize.milp`) läuft nur in den Tests und steht deshalb in `requirements-dev.txt`, nicht in `requirements.txt`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (fünf Experimente, Handbeispiel, Tab Reihenfolge, Grenzen, Formeln) |
| `wde_constants.py` | Vehikel, Messraster, Presets |
| `wde_rng.py` | SplitMix64 (Vorlage der Website, plattformfest) |
| `wde_knapsack.py` | Instanz, dynamische Programmierung, Status, Wertschwelle, Statustabelle, Aufzählung |
| `wde_rules.py` | Greedy und Regel S |
| `wde_schedule.py` | Reihenfolge (Aufzählung, Smith, Dispatch, geschlossene Gegenfakten) |
| `wde_estimates.py` | Schätzwert-Falle |
| `wde_ensemble.py` | Ensemble-Messung (erzeugt das Raster) |
| `wde_evaluation.py` | Aggregation über Ziehungen, Verdikte |
| `wde_visualization.py`, `wde_layout.py`, `wde_presets.py` | Abbildungen, fester Kopf, Permalink und Presets |
| `data/wde_vorgerechnet.json` | eingefrorenes Raster (fünf Ziehungen je Zelle) |
| `tools/wde_make_vorgerechnet.py` | rechnet das Raster, `--check` prüft es |
| `tests/` | Tests (siehe „Verifikation“) |

## Bewusst nicht enthalten / nicht gemessen

🚧 Das Folgestück „Warum nicht? Kontext und Unzulässigkeit“ (nächster Kontext im Sinn von Forel u. a., irreduzible unzulässige Teilsysteme, Löschfilter); lineare Programme (Verweis auf `lp-dualitaet-demo`); größere Instanzen und Zahlenskalen über 10; ein zweites, wertabhängiges oder verzerrtes Rauschmodell für die Schätzwert-Falle; Menschenstudien zur Lesbarkeit.

## Quellen

* Forel, Parmentier, Vidal (ICML 2023), Kurtz, Birbil, den Hertog (2024), Korikov, Beck (CP 2021), Lera-Leri u. a. (X-MILP 2025): **nur Zusammenfassungen über einen Quellenbericht gelesen (Stufe A)**; die Rechnungen dieser Demo sind eigene und keine Nachrechnung dieser Arbeiten. Čyras u. a. (AAAI 2019): Stufe S. Smith-Regel (Vertauschungsargument) und Rucksack-Tabelle sind Lehrbuch (siehe `wspt-demo`, `dynamic-programming-demo`). Vorbilder für „bekannte Wahrheit“: XAI-Bench, OpenXAI, Hooker (nicht Teil dieser Demo).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/wde_make_vorgerechnet.py --check     # Raster gegen den Quelltext prüfen und je eine Zelle neu rechnen
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
