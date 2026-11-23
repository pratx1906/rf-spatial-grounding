"""CPU rule-based rewording backend for scripts/paraphrase.py (--backend rules). Deterministic given the rng.

NOT human-written and NOT model-written: hand-made regex rewrites of the template wordings (sentence restructuring,
synonyms, operator-style framing), each output still checked by paraphrase.ok (protected spans, no echoed fact list,
meaning-critical words kept). Used for the labelled proxy set eval/frozen/para_v1 until human questions exist (P10).
"""

from __future__ import annotations

import random
import re

T = r"(trk:\w+)"
TS = r"t=(\d+(?:\.\d+)?) s"

# (pattern, [replacement templates]); groups are re-inserted verbatim
RULES: list[tuple[str, list[str]]] = [
    # replay
    (rf"^When was {T} first detected\?$", [r"At what time did we first pick up \1?", r"\1: when did it first show up?",
                                         r"What was the first detection time for \1?"]),
    (rf"^When was {T} last seen\?$", [r"What is the latest time \1 was seen?", r"\1 - when did we last see it?"]),
    (rf"^What is the last time {T} was tracked\?$", [r"When did we last have a track on \1?"]),
    (rf"^When did {T} change frequency band\?$", [r"At what time did \1 switch to another band?",
                                                  r"\1 hopped bands at some point. When?"]),
    (rf"^When did {T} start loitering\?$", [r"At what time did \1 start to loiter?", r"\1: when did it begin to loiter?"]),
    (rf"^When did {T} leave (.+)\?$", [r"At what time did \1 exit \2?", r"\1 left \2 at what time?"]),
    (rf"^When did {T} first enter (.+)\?$", [r"At what time did \1 first enter \2?", r"\1: first time it entered \2?"]),
    (rf"^At what time did {T} cross into (.+)\?$", [r"When did \1 enter \2?", r"\1 crossed into \2 - when?"]),
    (r"^Which tracks entered (.+) between (t=\d+ s) and (t=\d+ s)\?$",
     [r"Between \2 and \3, which tracks entered \1?", r"List the tracks that entered \1 between \2 and \3."]),
    (r"^Which track entered (.+) first, and when\?$", [r"What was the first track to enter \1, and at what time?"]),
    (r"^Who was the first to cross into (.+)\?$", [r"Which track entered \1 first?", r"First track that entered \1?"]),
    # operator query
    (rf"^Is {T} a genuine drone\?$", [r"Is \1 a real drone or something else?", r"Can you confirm \1 is a genuine UAV?"]),
    (rf"^Should we treat {T} as a real UAV\?$", [r"Is \1 a real drone?", r"Real UAV or not: \1?"]),
    (rf"^What kind of link is {T} transmitting\?$", [r"What type of radio link does \1 use?"]),
    (rf"^Identify the signal type of {T}\.$", [r"What signal type is \1 sending?", r"Tell me the link type of \1."]),
    (rf"^What radio protocol is {T} using\?$", [r"Which radio protocol does \1 use?"]),
    (rf"^Which restricted zone is {T} in at {TS}\?$", [r"At t=\2 s, which restricted zone contains \1?",
                                                       r"\1 at t=\2 s: which restricted zone is it in?"]),
    (rf"^At {TS}, {T} is inside which restricted zone\?$", [r"Which restricted zone is \2 in at t=\1 s?"]),
    (rf"^Is any drone inside (.+) at {TS}\? List the tracks\.$",
     [r"Do we have drones in \1 at t=\2 s? Which tracks?", r"At t=\2 s, are there drones inside \1? List them."]),
    (rf"^Any UAV within (.+) at {TS}\? Give the track IDs\.$",
     [r"Are there UAVs in \1 at t=\2 s? Track IDs please.", r"At t=\2 s, any drones within \1? Which tracks?"]),
    (rf"^Count the drone tracks in (.+) at {TS}\.$", [r"How many drone tracks are in \1 at t=\2 s?"]),
    (rf"^How many drones are inside (.+) at {TS}\? List them\.$", [r"At t=\2 s, how many drones are in \1? Which ones?"]),
    (rf"^Which drone is closest to (.+) at {TS}\?$", [r"At t=\2 s, which drone is closest to \1?",
                                                      r"What is the closest drone to \1 at t=\2 s?"]),
    (rf"^Nearest UAV to (.+) at {TS}\?$", [r"Which UAV is nearest to \1 at t=\2 s?"]),
    (rf"^List drone tracks on (.+) at {TS}\.$", [r"Which drone tracks are on \1 at t=\2 s?"]),
    (rf"^Which drones are transmitting on (.+) at {TS}\?$", [r"At t=\2 s, which drones transmit on \1?"]),
    (rf"^Does sensor (S[1-5]) have line of sight to {T} at {TS}\?$",
     [r"At t=\3 s, does \1 have a line of sight to \2?", r"Can \1 see \2 with line of sight at t=\3 s?"]),
    (rf"^Is (S[1-5])'s view of {T} at {TS} at risk of multipath ghosts\?$",
     [r"At t=\3 s, could \1 be seeing multipath ghosts of \2?"]),
    (rf"^Could sensor (S[1-5]) be seeing a ghost of {T} at {TS}\?$", [r"At t=\3 s, is \1 possibly seeing a ghost of \2?"]),
    (rf"^Which tracks fly in the same swarm as {T}\? List all members\.$",
     [r"List every track in the same swarm as \1.", r"\1 is in a swarm - which tracks are the members?"]),
    (r"^At (S[1-5]), which band has the highest noise floor rise\?$", [r"Which band shows the highest noise floor rise at \1?"]),
    # df
    (rf"^Locate the emitter of {T} at {TS} from the DF bearings\.$",
     [r"Using the DF bearings, where is the emitter of \1 at t=\2 s?", r"Give me a position for \1 at t=\2 s from the DF bearings."]),
    (rf"^Where exactly is {T} at {TS}\? Use the bearings and the site geometry\.$",
     [r"Using the bearings and site geometry, pin down \1 at t=\2 s.", r"Exact position of \1 at t=\2 s? Use bearings plus site geometry."]),
    (rf"^Locate {T} at {TS} with the bearings from (S[1-5]) and (S[1-5]) only\.$",
     [r"With only the \3 and \4 bearings, where is \1 at t=\2 s?"]),
    (rf"^Using only sensors (S[1-5]) and (S[1-5]), where is {T} at {TS}\?$",
     [r"Where is \3 at t=\4 s, using the bearings from \1 and \2 only?"]),
    (rf"^At {TS}, sensor (S[1-5]) reports (\d+) bearings for track {T}\. Which one is the direct path, and which are reflections\?$",
     [r"\2 has \3 bearings on \4 at t=\1 s. Which is the direct path and which are reflections?"]),
    (rf"^Sensor (S[1-5]) has (\d+) bearings on {T} at {TS}\. Which is direct\?$",
     [r"At t=\4 s, \1 shows \2 bearings on \3. Which one is direct?"]),
    (rf"^At {TS}, which of the DF bearings on {T} are direct paths and which are reflections\?$",
     [r"Split the DF bearings on \2 at t=\1 s into direct paths and reflections."]),
    (rf"^At {TS}, reject the ghost bearings on {T}; keep the direct ones\.$",
     [r"Drop the ghost bearings on \2 at t=\1 s and keep only the direct ones."]),
    # drone search
    (rf"^Fly the DF drone to refine the position of {T}\.(.*)$", [r"Send the DF drone to get a better fix on \1.\2"]),
    (rf"^Use the drone-borne DF to localise {T} as precisely as you can\.(.*)$",
     [r"Get the most precise position of \1 you can with the drone-borne DF.\2"]),
    (rf"^Task D1 to pin down {T}; report its position and uncertainty\.(.*)$",
     [r"Send D1 to pin down \1, then give its position and uncertainty.\2"]),
    # siting
    (r"^Where should we put one more DF node to improve (.+) coverage over (.+)\?$",
     [r"Best place for one more DF node to improve \1 coverage over \2?"]),
    (r"^Suggest two mount sites for extra DF nodes covering (.+) on (.+)\.$",
     [r"Which two mount sites would you pick for extra DF nodes covering \1 on \2?"]),
    # collateral
    (r"^If we jam (.+) from sensor (S[1-5]) at (\d+) dBm, which of our own DF sensors would be blinded\?$",
     [r"Jamming \1 from \2 at \3 dBm: which of our own DF sensors get blinded?"]),
    (r"^Which protected assets would a (\d+) dBm (.+) jammer at sensor (S[1-5]) disturb above (-\d+) dBm\?$",
     [r"A \1 dBm \2 jammer at sensor \3: which protected assets would it disturb above \4 dBm?"]),
    # unanswerable
    (rf"^Where was {T} at {TS}\?$", [r"What was the position of \1 at t=\2 s?"]),
    (rf"^Where is {T} now\?$", [r"Current position of \1?"]),
]
PREFIX = ["", "", "Quick one: ", "Operator here. ", "Need this now: "]


def reword(question: str, rng: random.Random) -> str | None:
    """One rule-based rewording, or None when no rule applies."""
    q = re.sub(r"\bthe the\b", "the", question)
    for pat, reps in RULES:
        if re.match(pat, q):
            out = re.sub(pat, rng.choice(reps), q)
            pre = rng.choice(PREFIX)
            lower = pre.endswith(": ") and not out.startswith(("trk:", "S1", "S2", "S3", "S4", "S5", "D1"))
            return pre + (out[0].lower() + out[1:] if lower else out)
    return None
