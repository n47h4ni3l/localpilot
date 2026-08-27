from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("output/pdf/atlas-of-human-emotions.pdf")


FAMILIES = [
    (
        "Enjoyment and ease",
        colors.HexColor("#F3A712"),
        "States that signal reward, safety, successful completion, play, or satisfying contact with life.",
        [
            ("Joy", "A broad sense that life is going well in the present moment.", "It often feels expansive, warm, energetic, and socially open. Joy invites savoring, play, sharing, and continued engagement with whatever is nourishing.", "Joy is wider and more enduring than delight, but more activated than contentment."),
            ("Delight", "A bright burst of pleasure produced by something especially pleasing.", "Attention narrows happily toward the object, event, or person. Smiling, laughter, leaning closer, and an urge to repeat or share the experience are common.", "Delight is sharper and more object-focused than general joy."),
            ("Amusement", "Pleasure arising from humor, incongruity, or playful violation.", "The body may release tension through smiling or laughter. Its action tendency is to continue the joke, bond through play, or reinterpret something rigidly held.", "Amusement can coexist with surprise; unlike contempt, it need not place another person beneath us."),
            ("Excitement", "High-energy pleasure directed toward an anticipated or unfolding event.", "It brings physiological activation, quickened attention, and an urge to approach. The same arousal can resemble anxiety when the outcome feels uncertain.", "Excitement expects gain; anxiety emphasizes possible threat."),
            ("Exhilaration", "Intense, liberating enjoyment accompanied by heightened vitality.", "People may feel powerful, fast, vividly alive, and temporarily less constrained. It often follows challenge, movement, risk, or a major success.", "Exhilaration is more intense and bodily than excitement, and less settled than triumph."),
            ("Contentment", "Quiet satisfaction with present conditions and no urgent need for change.", "Breathing and attention often soften. The impulse is to remain, savor, and conserve rather than chase or defend.", "Contentment is lower in arousal than joy and does not require the stillness associated with serenity."),
            ("Serenity", "Deep calm marked by inner steadiness and low conflict.", "The person may experience spacious attention, bodily ease, and acceptance of what is present. It supports reflection and patient contact with complexity.", "Serenity is more stable and encompassing than simple relaxation."),
            ("Relief", "Pleasure and release when a threat, burden, or uncertainty ends.", "Muscular tension drops, breathing changes, and attention disengages from monitoring danger. Relief can be intense even when nothing positively rewarding has occurred.", "Relief is defined by the removal of something aversive, unlike joy, which can arise from a positive gain."),
        ],
    ),
    (
        "Attachment and connection",
        colors.HexColor("#D95D8A"),
        "States that organize care, closeness, mutual reliance, social safety, and the wish to preserve valued bonds.",
        [
            ("Love", "A durable orientation of care, value, and attachment toward another being, group, place, ideal, or self.", "Love can contain warmth, longing, protectiveness, joy, grief, desire, duty, and deliberate commitment. Its central tendency is to support the beloved's continued existence and flourishing.", "Love is broader than affection and can persist even when momentary warmth is absent."),
            ("Affection", "Gentle fondness expressed through warm attention and friendly closeness.", "It often brings softness in voice and posture, a wish to touch or reassure, and pleasure in familiar presence. Affection can be brief or enduring.", "Affection is lighter and less totalizing than love."),
            ("Tenderness", "Protective softness toward perceived vulnerability or preciousness.", "The person becomes careful, attentive, and moved to comfort without overwhelming. Tenderness often slows behavior and lowers interpersonal threat.", "Tenderness centers fragility; compassion centers suffering and a wish to alleviate it."),
            ("Trust", "A felt willingness to rely on another person, system, or future outcome.", "Vigilance decreases and cooperation becomes easier because harmful betrayal is judged less likely. Trust always contains some exposure to risk.", "Confidence can concern one's own capacity; trust usually concerns reliance beyond oneself."),
            ("Belonging", "The felt sense of being accepted as a legitimate part of a relationship or group.", "It reduces social monitoring and supports participation, authenticity, and mutual obligation. Its absence can make neutral social cues feel threatening.", "Belonging is social inclusion; intimacy is depth of mutual access."),
            ("Intimacy", "Closeness created through mutual knowledge, vulnerability, responsiveness, and safety.", "It can be emotional, intellectual, physical, or spiritual. The impulse is to reveal, receive, and remain present with what is personally significant.", "Intimacy requires access and responsiveness, while familiarity alone may not."),
            ("Compassion", "Concern for suffering joined with a wish to reduce it.", "Compassion can feel warm and painful at once. It directs attention toward need while preserving enough steadiness to help effectively.", "Empathy shares or understands another's state; compassion adds care and an alleviating motive."),
            ("Gratitude", "Appreciation for a benefit understood as valuable and not entirely self-produced.", "It highlights interdependence and often motivates acknowledgment, reciprocity, stewardship, or generosity. Gratitude can be directed toward people, circumstance, nature, or life itself.", "Gratitude recognizes received good; admiration recognizes perceived excellence."),
        ],
    ),
    (
        "Interest, discovery, and meaning",
        colors.HexColor("#2A9D8F"),
        "States that orient attention toward novelty, complexity, learning, beauty, and patterns larger than the self.",
        [
            ("Curiosity", "A desire to reduce an information gap through exploration.", "It produces questioning, scanning, experimentation, and tolerance for temporary uncertainty. Healthy curiosity can coexist with caution and does not require immediate usefulness.", "Interest can be sustained enjoyment of a topic; curiosity is pulled by what is not yet known."),
            ("Interest", "Sustained positive engagement with an object, idea, person, or activity.", "Attention becomes easier to maintain and effort feels worthwhile. Interest supports learning by making detail feel salient rather than burdensome.", "Interest may remain after a question is answered, whereas curiosity often peaks around a gap."),
            ("Fascination", "Powerful attraction that repeatedly captures attention.", "The object feels rich, unusual, or difficult to exhaust. Fascination can deepen expertise, but it can also narrow awareness of competing needs.", "Fascination is more absorbing and less voluntary than ordinary interest."),
            ("Wonder", "Open, delighted perplexity before something striking or newly perceived.", "It combines not-knowing with receptivity rather than urgent control. Wonder invites contemplation, imaginative possibility, and renewed attention to familiar things.", "Wonder is gentler and more exploratory than awe."),
            ("Awe", "A response to perceived vastness that strains existing mental frames.", "The self may feel smaller, time may seem altered, and attention becomes panoramic. Awe can be pleasurable, frightening, or both.", "Awe requires perceived vastness and accommodation; wonder can arise from small or intimate details."),
            ("Inspiration", "A felt awakening to possibility that motivates expression or action.", "An example, idea, or encounter seems to reveal a higher standard or new path. Energy rises because the person can imagine becoming or making something different.", "Admiration can stop at appreciation; inspiration includes an impulse to embody or create."),
            ("Absorption", "Deep attentional immersion in which competing concerns recede.", "Time awareness may diminish and action can feel fluent. Absorption is not always pleasant, but it often supports learning, craft, play, and contemplation.", "Flow includes a good challenge-skill fit and feedback; absorption is the narrower experience of immersion."),
            ("Surprise", "A rapid orienting response when events violate expectation.", "Attention interrupts, eyes may widen, and the mind updates its model of what is happening. Surprise is brief and quickly blends into fear, joy, confusion, anger, or interest.", "Surprise is defined by unexpectedness, not by positive or negative value."),
        ],
    ),
    (
        "Hope, agency, and achievement",
        colors.HexColor("#5B8DEF"),
        "States that organize movement toward desired futures, effective action, challenge, and earned accomplishment.",
        [
            ("Hope", "A positive orientation toward a desired future that remains uncertain but possible.", "Hope supports persistence by holding both a valued goal and some path, however incomplete, toward it. It can coexist with grief, fear, and realism.", "Optimism expects favorable outcomes; hope can survive without high probability."),
            ("Anticipation", "Forward-looking activation as an expected event approaches.", "Attention rehearses possibilities and the body prepares to respond. Anticipation may be pleasant, unpleasant, or mixed depending on the forecast.", "Excitement is positively valenced; anticipation is neutral about whether the coming event is wanted."),
            ("Confidence", "A felt expectation that one's understanding or capability is adequate.", "It reduces hesitation and supports decisive action. Well-calibrated confidence remains responsive to evidence; brittle confidence defends itself from correction.", "Confidence concerns perceived capability, while courage concerns action despite fear."),
            ("Determination", "Firm commitment to continue toward a goal despite difficulty.", "Attention narrows around obstacles, effort mobilizes, and distractions lose priority. Determination can be adaptive persistence or rigid overcommitment.", "Motivation supplies energy; determination adds resolve in the face of resistance."),
            ("Courage", "Willing action in service of value despite fear, danger, or vulnerability.", "Fear may remain fully present while behavior aligns with principle or care. Courage therefore cannot be inferred from calm appearance alone.", "Fearlessness lacks fear; courage is defined by how fear is carried."),
            ("Empowerment", "A felt increase in agency, legitimacy, and capacity to influence outcomes.", "Posture, voice, and willingness to act may expand. Empowerment is strongest when supported by real resources and permission, not only positive self-talk.", "Confidence is a belief about competence; empowerment includes access to action and authority."),
            ("Pride", "Positive self-evaluation linked to an achievement, identity, relationship, or valued standard.", "It can straighten posture, invite visibility, and consolidate learning about effective effort. Healthy pride remains specific and compatible with respect for others.", "Pride values the self or its work; arrogance asserts superiority and resists proportion."),
            ("Triumph", "Intense satisfaction after overcoming a significant obstacle or opponent.", "It combines relief, pride, high activation, and a sense of restored power. The impulse is often to celebrate, display, or mark the victory.", "Triumph is event-bound and adversarial or obstacle-focused, unlike quieter accomplishment."),
        ],
    ),
    (
        "Threat, uncertainty, and fear",
        colors.HexColor("#735CDD"),
        "States that detect possible harm, prepare protection, and allocate attention to uncertain or immediate danger.",
        [
            ("Unease", "Low-intensity discomfort signaling that something may be off.", "It can appear as subtle tension, scanning, or reluctance without a clear story. Unease is useful as a cue to inspect rather than proof that danger exists.", "Unease is less defined and less activated than fear."),
            ("Worry", "Repetitive thought about possible negative outcomes and how to prevent them.", "It attempts problem solving through mental rehearsal, often without reaching closure. Worry can prepare action, but looping worry consumes attention without adding evidence.", "Worry is primarily verbal-cognitive; anxiety includes a broader bodily state."),
            ("Anxiety", "Apprehensive arousal in response to uncertain, anticipated, or diffuse threat.", "The body prepares for danger while attention searches for what might go wrong. It can sharpen preparation or narrow thought and amplify ambiguous cues.", "Fear usually has a more immediate object; anxiety often concerns possibility."),
            ("Fear", "A protective response to perceived danger.", "Heart rate, breathing, attention, and muscle readiness change to support escape, freezing, concealment, defense, or help-seeking. Fear is not evidence that the threat estimate is accurate.", "Fear is the broad state; alarm is its sudden orienting surge."),
            ("Dread", "Heavy anticipation of an aversive event felt as difficult to avoid.", "The future seems to cast a shadow backward into the present. Dread often combines fear, helplessness, and sustained mental simulation.", "Dread is slower and more future-saturated than acute fear."),
            ("Alarm", "A sudden spike of threat detection demanding immediate orientation.", "The body interrupts ongoing activity and prepares rapid action before full interpretation. It is brief unless the danger remains unresolved.", "Alarm is the initial emergency signal; panic is a more overwhelming cascade."),
            ("Panic", "Intense fear accompanied by a felt loss of control or imminent catastrophe.", "Sensations can include racing heart, breathlessness, dizziness, unreality, and urgent escape. The sensations themselves may become interpreted as further danger.", "Panic is distinguished by overwhelming intensity and catastrophic interpretation."),
            ("Vulnerability", "Awareness of exposure to hurt, loss, judgment, or dependence.", "It can feel tender, unsafe, honest, or connective depending on context and trust. Vulnerability often creates simultaneous impulses to hide and to seek care.", "Vulnerability is a condition felt emotionally; helplessness adds perceived inability to respond."),
        ],
    ),
    (
        "Boundaries, obstruction, and anger",
        colors.HexColor("#E4572E"),
        "States that register blockage, violation, unfairness, coercion, or the need to restore agency and limits.",
        [
            ("Irritation", "Low-intensity anger at a recurring friction or minor intrusion.", "Attention repeatedly returns to the disturbance and patience contracts. The impulse is to remove the annoyance, correct it, or create distance.", "Irritation is smaller and more local than anger."),
            ("Frustration", "Activation produced when progress toward a goal is blocked.", "Effort rises, strategies may change, and the obstacle becomes highly salient. Prolonged frustration can shift into anger, resignation, or creative adaptation.", "Frustration centers obstruction; anger often includes blame or violation."),
            ("Anger", "Mobilizing displeasure in response to perceived wrong, threat, or blocked agency.", "Energy moves outward toward confrontation, correction, protection, or boundary-setting. Anger carries information about values but does not by itself justify the target or response.", "Anger is the family state; rage is extreme intensity and reduced flexibility."),
            ("Indignation", "Anger shaped by a judgment that something is unfair or morally wrong.", "It motivates protest, accountability, and defense of standards or victims. Because it feels principled, it can obscure uncertainty about facts or proportionality.", "Indignation is moralized anger; frustration need not involve wrongdoing."),
            ("Resentment", "Enduring anger about perceived mistreatment, imbalance, or unacknowledged sacrifice.", "The grievance is mentally preserved because repair, recognition, or boundary change feels incomplete. It may protect memory of harm while also prolonging attachment to it.", "Resentment persists over time; anger can be immediate and brief."),
            ("Rage", "Extremely intense anger with urgent pressure toward forceful action.", "Attention narrows, bodily power surges, and nuance can collapse. Rage signals a severe perceived threat or violation but creates high risk of disproportionate behavior.", "Rage is an intensity state, not evidence that violence is necessary or justified."),
            ("Impatience", "Agitated resistance to delay, slowness, or unmet timing expectations.", "The person wants acceleration and may interrupt, rush, or abandon process. It can reveal urgency, entitlement, anxiety, or simple time scarcity.", "Impatience concerns pace; frustration concerns blocked progress more generally."),
            ("Defiance", "Energized refusal to submit to control, demand, or expectation.", "It strengthens autonomy and willingness to bear consequences. Defiance can defend dignity or become reflexive opposition that ignores shared goals.", "Assertiveness communicates a boundary; defiance emphasizes resistance to authority."),
        ],
    ),
    (
        "Aversion, status, and rivalry",
        colors.HexColor("#6B8E23"),
        "States that reject contamination or degradation, monitor hostile intent, and compare access to valued bonds or resources.",
        [
            ("Disgust", "Aversion that marks something as contaminating, degrading, or unfit for contact.", "The face and body may recoil, nausea may arise, and the impulse is to expel, avoid, or cleanse. Disgust can protect health but can also be misapplied socially.", "Disgust rejects contact; fear prioritizes protection from danger."),
            ("Revulsion", "Intense disgust with a powerful urge to withdraw or reject.", "The stimulus feels intolerable at close range, physically or morally. Revulsion can remain long after the immediate encounter through vivid memory.", "Revulsion is a stronger, more engulfing form of disgust."),
            ("Contempt", "Cold negative evaluation that places another person or group beneath respect.", "It often reduces curiosity and licenses dismissal, mockery, or distance. Because it denies equal standing, contempt is especially corrosive in close relationships.", "Anger may still seek engagement or repair; contempt tends to devalue the person."),
            ("Suspicion", "Protective doubt about another's motives, claims, or reliability.", "Attention searches for inconsistency and withholds trust pending evidence. Calibrated suspicion supports safety; unchecked suspicion interprets ambiguity as confirmation.", "Skepticism evaluates a claim; suspicion often evaluates intent."),
            ("Envy", "Painful comparison when another possesses something one desires.", "Attention fixes on the gap in status, ability, opportunity, or possession. Envy can motivate growth, withdrawal, devaluation, or a wish that the advantage disappear.", "Envy concerns what another has; jealousy concerns threatened possession of a bond or position."),
            ("Jealousy", "Threat response to the possible loss of a valued relationship or standing to a rival.", "It combines fear, anger, vigilance, and attachment. The person may seek reassurance, closeness, control, competition, or withdrawal.", "Jealousy is usually triangular; envy requires only self and comparison target."),
            ("Moral outrage", "High-arousal anger and disgust at a perceived serious violation of shared values.", "It can mobilize collective action, punishment, and public signaling. Its social power makes verification and proportionality especially important.", "Moral outrage is outward and norm-focused; guilt is inward and responsibility-focused."),
            ("Schadenfreude", "Pleasure at another's misfortune, often when it restores status balance or feels deserved.", "The pleasure may be hidden because it conflicts with compassion or social norms. It can reveal rivalry, resentment, justice judgments, or relief that harm fell elsewhere.", "It differs from simple amusement because another's setback is central to the pleasure."),
        ],
    ),
    (
        "Loss, separation, and sadness",
        colors.HexColor("#4974A5"),
        "States that register irreversible change, unmet need, separation, and the reduced possibility of a valued future.",
        [
            ("Sadness", "Low-energy pain in response to loss, disappointment, or diminished possibility.", "It slows action, turns attention inward, and may invite rest, meaning-making, or support. Sadness can clarify what mattered without dictating what must happen next.", "Sadness is broad; grief is organized around significant loss and adaptation."),
            ("Sorrow", "Deep, sustained sadness with solemn awareness of suffering or loss.", "It often feels weighty and reflective rather than chaotic. Sorrow may be personal or extend empathically toward others and the world.", "Sorrow is usually deeper and more dignified in tone than ordinary sadness."),
            ("Grief", "The changing emotional process of adapting to significant loss.", "It may include yearning, numbness, anger, guilt, relief, disorientation, love, and moments of joy. Grief is not one steady feeling and does not follow a universal timetable.", "Bereavement names a circumstance; grief names the lived adaptation."),
            ("Heartbreak", "Acute pain associated with rupture, rejection, or impossibility in a valued bond.", "The attachment system continues seeking what is no longer available, producing yearning, intrusive memory, bodily ache, and identity disruption.", "Heartbreak is attachment-centered; grief can follow any profound loss."),
            ("Loneliness", "Painful awareness that desired social connection is absent or insufficient.", "It increases attention to social cues and motivates contact, but threat sensitivity can also make connection harder. A person can be lonely in a crowd or content while alone.", "Solitude is a condition; loneliness is an unmet relational need."),
            ("Homesickness", "Longing and distress caused by separation from a familiar place, people, routines, or identity context.", "Memories become vivid and the unfamiliar environment may feel emotionally thin. It combines attachment, nostalgia, anxiety, and dislocation.", "Homesickness is place-and-belonging specific, unlike generalized loneliness."),
            ("Disappointment", "Sadness when reality falls short of an expectation or hoped-for outcome.", "Energy drops as the imagined future is revised. It can lead to learning, blame, resignation, or renewed planning depending on perceived control.", "Regret focuses on one's choices; disappointment can occur without personal responsibility."),
            ("Melancholy", "Reflective, often diffuse sadness that can contain beauty, tenderness, or contemplation.", "It tends to be lower in urgency than acute grief and may not have one clear cause. Melancholy can deepen attention while also reducing momentum.", "Melancholy is mood-like and atmospheric; sorrow is more directly tied to suffering or loss."),
        ],
    ),
    (
        "Self-evaluation and exposure",
        colors.HexColor("#9C6644"),
        "States that compare the self with personal standards, social expectations, responsibility, and perceived public judgment.",
        [
            ("Embarrassment", "Discomfort after a social misstep or unwanted exposure that is usually limited and repairable.", "Attention turns toward how one appears, and the impulse may be to smile, hide, apologize, or restore smooth interaction. It often communicates recognition of a norm without global self-condemnation.", "Embarrassment concerns a moment; shame can condemn the whole self."),
            ("Shame", "Painful sense that the self is defective, exposed, or unworthy of belonging.", "The body may collapse, hide, freeze, or seek disappearance. Shame can encourage repair when proportionate, but global shame attacks identity rather than behavior.", "Guilt says 'I did something wrong'; shame says 'something is wrong with me.'"),
            ("Guilt", "Distress linked to believing one has violated a value or harmed someone.", "It directs attention to responsibility and often motivates confession, apology, restitution, or changed behavior. Guilt is most constructive when specific and proportionate.", "Guilt evaluates an action; shame evaluates the person."),
            ("Remorse", "Deep guilt joined with empathic understanding of harm and sincere wish for repair.", "The event remains emotionally alive because its consequences matter. Remorse accepts responsibility more fully than defensive regret.", "Regret may concern a bad outcome for oneself; remorse centers wrongdoing and its impact on others."),
            ("Regret", "Painful comparison between what happened and a better outcome imagined from another choice.", "The mind simulates alternatives and extracts lessons about agency. Regret can guide future decisions or become repetitive counterfactual punishment.", "Disappointment needs an unmet expectation; regret usually includes perceived choice."),
            ("Humiliation", "Pain and anger from being forcibly lowered, exposed, or stripped of dignity before others.", "It combines shame with perceived unjust social domination and may motivate hiding, retaliation, or restoration of status. The experience depends heavily on power and audience.", "Embarrassment can be self-generated and mild; humiliation is imposed and degrading."),
            ("Inadequacy", "A felt shortfall between one's capacity and a valued demand or comparison standard.", "It can motivate learning, avoidance, envy, or despair depending on whether growth seems possible. The feeling is information about appraisal, not proof of objective worth.", "Insecurity anticipates unstable acceptance; inadequacy focuses on not being enough."),
            ("Self-respect", "Steady valuing of one's dignity, needs, limits, and moral agency.", "It supports boundaries and responsibility without requiring superiority. Self-respect can feel quiet rather than triumphant and may demand difficult choices.", "Pride often follows achievement; self-respect concerns baseline worth and integrity."),
        ],
    ),
    (
        "Low activation, depletion, and stuckness",
        colors.HexColor("#6C757D"),
        "States that reduce action, signal depleted reward or energy, and sometimes protect against overload.",
        [
            ("Boredom", "Aversive under-engagement when available activity lacks meaning, challenge, or novelty.", "Attention wanders and seeks change. Boredom can stimulate exploration, but chronic boredom may lead to impulsive stimulation or disengagement.", "Apathy lacks motivation itself; boredom contains motivation to be engaged differently."),
            ("Apathy", "Reduced interest, motivation, and emotional investment.", "Goals lose pull and initiation becomes difficult even when action is possible. Apathy may reflect depletion, illness, protection, or loss of expected reward.", "Numbness is reduced feeling; apathy is reduced motivation and concern."),
            ("Numbness", "Diminished access to feeling, often after overload, shock, or prolonged stress.", "The world may seem distant, flat, or unreal. Numbness can be protective in the short term while obscuring needs and delaying processing.", "Calm retains contact and flexibility; numbness reduces emotional access."),
            ("Restlessness", "Unsettled activation without a satisfying direction.", "The body wants movement and attention rejects stillness, yet no option feels quite right. It can reflect anticipation, boredom, anxiety, excess energy, or constrained agency.", "Impatience has a target delay; restlessness may have no clear object."),
            ("Languishing", "A prolonged sense of stagnation, low vitality, and incomplete engagement with life.", "Functioning may continue while meaning and momentum feel muted. It is neither ordinary boredom nor necessarily a clinical disorder.", "Apathy is reduced motivation; languishing is a broader pattern of not flourishing."),
            ("Resignation", "Acceptance that further effort is unlikely to change an unwanted outcome.", "Struggle decreases, sometimes bringing relief and sometimes sadness. Resignation can be realistic conservation or premature surrender shaped by learned helplessness.", "Acceptance can preserve agency and values; resignation emphasizes relinquished influence."),
            ("Helplessness", "The felt inability to prevent harm, meet need, or alter an important situation.", "Energy may collapse into freezing, passivity, or desperate help-seeking. Repeated uncontrollability can generalize beyond the original context.", "Vulnerability means exposure; helplessness means perceived lack of effective response."),
            ("Despair", "Profound loss of hope in which valued futures feel closed.", "Motivation contracts because no meaningful path seems available. Despair requires care and connection because its certainty about the future can exceed the evidence.", "Sadness mourns loss; despair concludes that improvement or meaning is no longer possible."),
        ],
    ),
    (
        "Social appreciation and moral feeling",
        colors.HexColor("#7A5195"),
        "States that respond to another's excellence, suffering, virtue, authority, or social position.",
        [
            ("Admiration", "Positive regard for perceived excellence, skill, character, or achievement.", "Attention highlights what is exemplary and may motivate imitation, praise, or learning. Admiration can remain compatible with equality.", "Awe emphasizes vastness; admiration evaluates excellence."),
            ("Respect", "Recognition of another's standing, rights, competence, or boundaries.", "It shapes restraint and fair treatment even without liking or agreement. Respect can be interpersonal, role-based, earned, or grounded in basic dignity.", "Admiration is warm approval; respect can persist without warmth."),
            ("Reverence", "Deep respect infused with humility, solemnity, or sacred significance.", "Speech and movement may slow as the person protects what feels worthy of honor. Reverence can be religious or secular.", "Awe can overwhelm; reverence includes a relational posture of care and honor."),
            ("Elevation", "Warm uplift in response to witnessing moral beauty or exceptional goodness.", "The chest may feel open and the person may want to become kinder or act prosocially. It transforms admiration into moral aspiration.", "Inspiration can arise from any possibility; elevation is specifically elicited by virtue."),
            ("Empathy", "Affective or cognitive contact with another person's experience.", "One may resonate with the feeling, understand its perspective, or both. Empathy provides information but does not guarantee accuracy, agreement, or helpful action.", "Compassion adds care and the wish to reduce suffering."),
            ("Sympathy", "Concern for another's difficulty from a position that does not necessarily share the same feeling.", "It communicates alliance and acknowledgment, often motivating comfort. Its distance can be useful or can feel patronizing if it overlooks the other's agency.", "Empathy attempts to understand from within; sympathy feels concern from alongside."),
            ("Pity", "Sorrow for someone perceived as suffering and comparatively powerless.", "It may motivate aid, but the implied status difference can threaten dignity. Pity becomes humane when joined with respect and curiosity about actual needs.", "Compassion need not place helper above sufferer; pity often contains that asymmetry."),
            ("Social pride", "Positive emotion from identification with another person's or group's valued achievement.", "It strengthens shared identity and motivates celebration or continued contribution. It can support solidarity or slide into exclusionary superiority.", "Personal pride centers one's own achievement; social pride is vicarious or collective."),
        ],
    ),
    (
        "Mixed, temporal, and hard-to-name states",
        colors.HexColor("#4D908E"),
        "Blended states that combine opposing values, time perspectives, incomplete categories, and complex appraisals.",
        [
            ("Nostalgia", "Warm, often bittersweet longing for a personally meaningful past.", "Memory reconstructs belonging, identity, people, or places that feel distant. Nostalgia can restore continuity while also sharpening awareness that the past cannot be fully recovered.", "Homesickness longs for home during separation; nostalgia can concern any lost time or world."),
            ("Bittersweetness", "Simultaneous pleasure and sadness within the same meaningful experience.", "The person may savor more intensely because beauty and loss are both present. Neither pole cancels the other.", "Ambivalence involves competing evaluations; bittersweetness specifically blends positive feeling with sorrow."),
            ("Poignancy", "A piercing sense of meaning created by tenderness, beauty, vulnerability, and impermanence.", "It can bring tears without simple sadness and gratitude without simple joy. Attention slows around what matters because it will change or end.", "Poignancy is a meaning-laden moment; melancholy is a more diffuse mood."),
            ("Ambivalence", "Coexisting positive and negative evaluations of the same object or choice.", "Approach and avoidance alternate, making decisions effortful. Ambivalence can signal conflict, complexity, or insufficient information rather than weakness.", "Indecision is difficulty choosing; ambivalence is the emotional structure producing competing pulls."),
            ("Confusion", "Discomfort when available information does not fit a coherent model.", "Attention searches for missing distinctions, causal links, or rules. Confusion can motivate inquiry if the person feels capable, or withdrawal if overload dominates.", "Uncertainty is not knowing; confusion is difficulty organizing what is already present."),
            ("Uncertainty", "Awareness that an outcome, fact, identity, or course of action is not settled.", "It can feel open and curious or threatening and unstable. Tolerance of uncertainty allows evidence gathering without premature closure.", "Doubt questions a proposition; uncertainty describes incomplete determination more broadly."),
            ("Yearning", "Sustained, emotionally charged desire for something absent or not yet attainable.", "Attention repeatedly returns to the wanted person, place, state, or future. Yearning may energize pursuit or deepen pain when no path exists.", "Wanting can be brief and practical; yearning is enduring and identity-relevant."),
            ("Wistfulness", "Gentle, reflective longing tinged with sadness and acceptance.", "It looks toward an unrealized or vanished possibility without the full urgency of yearning. The state can feel tender rather than destabilizing.", "Wistfulness is lighter and more accepting than heartbreak or acute regret."),
        ],
    ),
]


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8DEE8"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 8 * mm, "Atlas of Human Emotions - LocalPilot library test edition")
    canvas.drawRightString(192 * mm, 8 * mm, str(document.page))
    canvas.restoreState()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "AtlasTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=30,
        leading=34,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#172033"),
        spaceAfter=8 * mm,
    )
    subtitle = ParagraphStyle(
        "AtlasSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475467"),
    )
    heading = ParagraphStyle(
        "FamilyHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#172033"),
        spaceAfter=4 * mm,
    )
    intro = ParagraphStyle(
        "FamilyIntro",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#475467"),
        spaceAfter=6 * mm,
    )
    card_title = ParagraphStyle(
        "CardTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#172033"),
        spaceAfter=2 * mm,
    )
    card_body = ParagraphStyle(
        "CardBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.5,
        textColor=colors.HexColor("#253046"),
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475467"),
    )
    index_style = ParagraphStyle(
        "Index",
        parent=styles["BodyText"],
        fontSize=8.6,
        leading=12.5,
        textColor=colors.HexColor("#344054"),
    )

    story = [
        Spacer(1, 35 * mm),
        Paragraph("ATLAS OF HUMAN EMOTIONS", title),
        Paragraph(
            "A practical phenomenological field guide for recognition, reflection, and LocalPilot library testing",
            subtitle,
        ),
        Spacer(1, 18 * mm),
        Table(
            [[Paragraph(
                "No finite list can literally contain every human emotion. Emotional vocabulary varies across cultures, languages, bodies, histories, and situations. This atlas offers a broad map of 88 commonly named states. Each entry describes a core appraisal, felt texture, action tendency, and useful distinction from nearby states. It is a reference, not a diagnosis.",
                intro,
            )]],
            colWidths=[150 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F4F7")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D0D5DD")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 7 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7 * mm),
            ]),
        ),
        Spacer(1, 20 * mm),
        Paragraph("Original synthesis prepared for the owner-managed LocalPilot library", subtitle),
        PageBreak(),
        Paragraph("How to use this atlas", heading),
        Paragraph(
            "Treat an emotion word as a hypothesis, not a verdict. Start with the situation, bodily changes, attention pattern, impulse, values, and social context. Several emotions can be active at once, and the same bodily arousal can acquire different meanings. Naming is most useful when it expands options rather than forcing experience into a box.",
            intro,
        ),
        Paragraph("A compact observation sequence", card_title),
        Paragraph(
            "1. Notice: What changed in body, attention, energy, and imagery?  2. Locate: What event, memory, forecast, or relationship became salient?  3. Appraise: What seems gained, lost, threatened, blocked, unfair, uncertain, or meaningful?  4. Name provisionally: Which two or three entries fit best?  5. Differentiate: What nearby state would imply a different need or action?  6. Choose: What response protects values without treating emotion as unquestionable evidence?",
            card_body,
        ),
        Spacer(1, 8 * mm),
        Paragraph("Family index", heading),
    ]
    for family_name, color, family_intro, entries in FAMILIES:
        names = ", ".join(entry[0] for entry in entries)
        story.append(
            Table(
                [["", Paragraph(f"<b>{family_name}</b><br/>{names}", index_style)]],
                colWidths=[5 * mm, 165 * mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (0, 0), color),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E4E7EC")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("LEFTPADDING", (1, 0), (1, 0), 4 * mm),
                ]),
            )
        )
        story.append(Spacer(1, 2 * mm))

    for family_name, color, family_intro, entries in FAMILIES:
        story.extend([PageBreak(), Paragraph(family_name, heading), Paragraph(family_intro, intro)])
        for name, core, texture, distinction in entries:
            card = [
                Paragraph(name, card_title),
                Paragraph(f"<b>Core:</b> {core}", card_body),
                Spacer(1, 1.5 * mm),
                Paragraph(f"<b>Texture and tendency:</b> {texture}", card_body),
                Spacer(1, 1.5 * mm),
                Paragraph(f"<b>Distinction:</b> {distinction}", card_body),
            ]
            table = Table(
                [[card]],
                colWidths=[170 * mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.7, color),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                    ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
                ]),
            )
            story.extend([table, Spacer(1, 3.5 * mm)])

    story.extend([
        PageBreak(),
        Paragraph("Closing note: emotions as information, not commands", heading),
        Paragraph(
            "Emotions are embodied appraisals that organize attention and action. They can reveal needs, values, predictions, memories, and social meanings, but they are not infallible measurements of the world. A mature response can honor the signal while checking the interpretation. Curiosity can ask what is missing. Compassion can make room for pain. Courage can act with fear present. Evidence can correct the story without erasing the experience.",
            intro,
        ),
        Paragraph(
            "For LocalPilot: passages from this atlas are source material. Quote or paraphrase them with a library citation, preserve uncertainty, and do not infer that reading a passage created durable memory or changed model weights.",
            small,
        ),
    ])

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Atlas of Human Emotions",
        author="LocalPilot project",
        subject="Owner-managed library test reference",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT.resolve()


if __name__ == "__main__":
    print(build())
