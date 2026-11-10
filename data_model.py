"""
Data model for psychiatric history-taking bot.
Defines sections, field configurations, and state constants.
"""

# Conversation states
LANGUAGE_SELECTION = "LANGUAGE_SELECTION"
COLLECTING_HISTORY = "COLLECTING_HISTORY"
CRISIS_MODE = "CRISIS_MODE"

# Section order
SECTIONS_ORDER = [
    "sociodemographic",
    "chief_complaints",
    "factors_of_illness",
    "hopi",
    "negative_history",
    "past_history",
    "family_history",
    "personal_history",
    # "mse"  ← DISABLED. A mental status exam cannot be conducted over a text
    # chat. Of its 22 fields, 11 require OBSERVING the patient (appearance,
    # psychomotor activity, speech, objective mood, affect quality/intensity/
    # mobility, thought stream/form, consciousness, intelligence) — the bot was
    # asking patients to self-describe their own affect and appearance, and
    # min_required=8 forced it to bank eight such "findings". That is how real
    # reports came to assert "the patient's affect and cognition appear intact"
    # about someone nobody had ever seen. The serial-7s test also loops forever
    # on a wrong answer, because impaired arithmetic is treated as a failed
    # extraction rather than as the clinical finding it actually is.
    #
    # The interview now ends after personal_history (get_next_section returns
    # None -> finish_session). SECTION_CONFIG["mse"] and
    # SECTION_CONVERSATION_TOPICS["mse"] are left defined below but are now
    # unreachable, so min_required=8 no longer applies. Re-enable by
    # uncommenting this line — but the observation-only fields and the
    # serial-7s handling must be fixed first.
]

# Detailed section configuration
SECTION_CONFIG = {
    "sociodemographic": {
        "display_name": "About You",
        "baseline_minutes": 4,
        "min_required": 11,
        "fields": [
            "name", "age", "sex", "education", "occupation", "ses",
            "marital_status", "religion", "residence",
            "mother_tongue", "address", "source_of_referral",
            "informant", "reliability_of_info", "family_type"
        ]
    },
    "chief_complaints": {
        "display_name": "Your Main Concerns",
        "baseline_minutes": 5,
        "min_required": 3,
        "fields": [
            "complaint_1", "complaint_2", "complaint_3",
            "complaint_4", "complaint_5",
            "duration_each", "impact_on_functioning"
        ]
    },
    "factors_of_illness": {
        "display_name": "Background & Triggers",
        "baseline_minutes": 4,
        "min_required": 3,
        "fields": [
            "predisposing_biological", "predisposing_psychological", "predisposing_social",
            "precipitating_biological", "precipitating_psychological", "precipitating_social",
            "perpetuating_biological", "perpetuating_psychological", "perpetuating_social",
            "mode_of_onset", "course_of_illness", "progress_of_illness"
        ]
    },
    "hopi": {
        "display_name": "Your Symptoms in Detail",
        "baseline_minutes": 15,
        "min_required": 6,
        "fields": [
            "symptom_chronology", "site", "onset_timing", "character",
            "radiation_or_spread", "associations", "time_course",
            "exacerbating_factors", "relieving_factors", "severity",
            "hospitalizations", "suicidality_assessed"
        ]
    },
    "negative_history": {
        "display_name": "Other Health Areas",
        "baseline_minutes": 5,
        "min_required": 4,
        "fields": [
            "cvs_negative", "respiratory_negative", "gi_negative",
            "neuro_negative", "genitourinary_negative",
            "musculoskeletal_negative", "endocrine_negative"
        ]
    },
    "past_history": {
        "display_name": "Past Health History",
        "baseline_minutes": 10,
        "min_required": 4,
        "fields": [
            "previous_psychiatric_episodes", "previous_diagnoses",
            "treatment_history", "medications_past", "hospitalizations_past",
            "medical_surgical_history", "drug_history_current", "allergies"
        ]
    },
    "family_history": {
        "display_name": "Family Background",
        "baseline_minutes": 10,
        "min_required": 6,
        "fields": [
            "family_members_list", "living_arrangement",
            "education_occupation_members", "functional_head",
            "nominal_head", "relationship_quality", "family_dynamics",
            "psychiatric_illness_in_family", "genetic_conditions", "support_system"
        ]
    },
    "personal_history": {
        "display_name": "Personal Life",
        "baseline_minutes": 10,
        "min_required": 6,
        "fields": [
            "birth_and_early_development", "developmental_milestones",
            "childhood_environment", "school_history",
            "peer_relationships_childhood", "sexual_development",
            "occupational_history", "marital_history",
            "reproductive_history", "substance_use_history",
            "forensic_history", "premorbid_personality"
        ]
    },
    "mse": {
        "display_name": "Your Current State",
        "baseline_minutes": 6,
        "min_required": 8,
        "fields": [
            "general_appearance_behaviour", "psychomotor_activity",
            "speech_characteristics", "mood_subjective", "mood_objective",
            "affect_quality", "affect_intensity", "affect_mobility",
            "thought_stream", "thought_form", "thought_content", "thought_possession",
            "perception", "consciousness", "attention_concentration",
            "orientation", "memory", "general_knowledge",
            "intelligence", "abstract_ability", "judgment", "insight"
        ]
    }
}

# Milestone update thresholds (percentage)
MILESTONE_THRESHOLDS = [25, 50, 75, 100]

# Progress update rate limit (seconds)
PROGRESS_UPDATE_MIN_INTERVAL_SECONDS = 30

# Default language detection fallback
DEFAULT_LANGUAGE = "en"

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "🇬🇧 English",
    "hi": "🇮🇳 हिंदी",
    "hinglish": "🇮🇳 Hinglish (Hindi in English)"
}

# ============================================================================
# TOPIC-BASED CONVERSATION SYSTEM
# Groups related fields into natural conversational topics instead of asking
# for fields sequentially. Each section flows naturally through topics.
# ============================================================================

SECTION_CONVERSATION_TOPICS = {
    "sociodemographic": [
        {
            "topic_id": "background_basics",
            "topic_name": "Basic Background",
            "natural_questions_en": [
                "Tell me a bit about yourself. How old are you and what is your gender (male/female/other)?",
                "Could you tell me your age and your gender?",
            ],
            "natural_questions_hi": [
                "Apne baare mein thoda bataiye. Aap kitne saal ke hain aur aapka gender kya hai (male/female/other)?",
                "Kya aap apni umar aur gender bata sakte hain?",
            ],
            "natural_questions_hinglish": [
                "Apne baare mein thoda batao. Aapki age kya hai aur aapka gender kya hai (male/female/other)?",
                "Aapki age aur gender kya hai?",
            ],
            # Targeted fallback questions used when the topic is only partially answered.
            # Keyed by the specific field still missing, so we never re-ask what's already known.
            "field_questions_en": {
                "age": "How old are you?",
                "sex": "And what's your gender, if you don't mind sharing?",
            },
            "field_questions_hi": {
                "age": "Aap kitne saal ke hain?",
                "sex": "Aur aapka gender kya hai, agar batana chahein?",
            },
            "field_questions_hinglish": {
                "age": "Aap kitne saal ke ho?",
                "sex": "Aur aapka gender kya hai, agar share karna chaho?",
            },
            "fields": ["age", "sex"],
            "min_required": 2,
        },
        {
            "topic_id": "work_education",
            "topic_name": "Education & Occupation",
            "natural_questions_en": [
                "What is your highest level of education? And what is your current occupation?",
                "Have you completed any formal education or vocational training? And are you currently working?",
            ],
            "natural_questions_hi": [
                "Aapki sabse zyada padhai kahan tak hui hai? Aur aap abhi kya kaam karte hain?",
            ],
            "natural_questions_hinglish": [
                "Aapki highest education kya hai? Aur current occupation kya hai?",
            ],
            "field_questions_en": {
                "education": "Are you currently enrolled in any educational programs or courses?",
                "occupation": "Are you currently working full-time, part-time, or are you unemployed?",
            },
            "field_questions_hi": {
                "education": "Kya aap abhi kisi padhai ya course mein enrolled hain?",
                "occupation": "Kya aap abhi full-time kaam karte hain, part-time, ya berozgaar hain?",
            },
            "field_questions_hinglish": {
                "education": "Kya aap abhi kisi educational program ya course mein enrolled ho?",
                "occupation": "Aap abhi full-time work karte ho, part-time, ya unemployed ho?",
            },
            "fields": ["education", "occupation"],
            "min_required": 2,
        },
        {
            "topic_id": "socioeconomic_living",
            "topic_name": "Living Situation & Income",
            "natural_questions_en": [
                "Where do you currently live — a city or a rural area?",
                "Do you live in a city or a village? And what is your monthly income?",
            ],
            "natural_questions_hi": [
                "Aap abhi kahan rehte hain — sheher mein ya gaon mein?",
                "Kya aap sheher mein rehte hain ya gaon mein? Aapki monthly income kitni hai?",
            ],
            "natural_questions_hinglish": [
                "Aap abhi kahan rehte ho — city ya rural area?",
                "Kya aap city mein rehte ho ya village mein? Aapki monthly income kitni hai?",
            ],
            "field_questions_en": {
                "residence": "Where do you currently live — a city or a rural area?",
                "ses": "What is your monthly income, or your total family monthly income (in INR)?",
                "address": "Could you provide your current address or general location?",
            },
            "field_questions_hi": {
                "residence": "Aap abhi kahan rehte hain — sheher mein ya gaon mein?",
                "ses": "Aapki ya parivar ki mahine ki aamdani kitni hai?",
                "address": "Kya aap apna abhi ka pata ya ilaka bata sakte hain?",
            },
            "field_questions_hinglish": {
                "residence": "Aap abhi kahan rehte ho — city ya rural area?",
                "ses": "Aapki ya family ki monthly income kitni hai?",
                "address": "Aapka current address ya area kya hai?",
            },
            "fields": ["residence", "ses", "address"],
            "min_required": 2,
        },
        {
            "topic_id": "personal_status",
            "topic_name": "Marital Status",
            "natural_questions_en": [
                "What is your current marital status — single, married, divorced, separated, or widowed?",
            ],
            "natural_questions_hi": [
                "Aapka marital status kya hai — single, married, divorced, separated ya widowed?",
            ],
            "natural_questions_hinglish": [
                "Aapka current marital status kya hai — single, married, divorced, separated ya widowed?",
            ],
            "field_questions_en": {
                "marital_status": "What is your current marital status — single, married, divorced, separated, or widowed?",
            },
            "field_questions_hi": {
                "marital_status": "Aapka marital status kya hai — single, married, divorced, separated ya widowed?",
            },
            "field_questions_hinglish": {
                "marital_status": "Aapka marital status kya hai?",
            },
            "fields": ["marital_status"],
            "min_required": 1,
        },
        {
            "topic_id": "cultural_background",
            "topic_name": "Religion & Mother Tongue",
            "natural_questions_en": [
                "Do you follow a particular religion or spiritual practice? If yes, which one?",
            ],
            "natural_questions_hi": [
                "Kya aap koi dharm ya adhyatmik practice follow karte hain? Agar haan, to kaun sa?",
            ],
            "natural_questions_hinglish": [
                "Kya aap koi religion ya spiritual practice follow karte ho? Kaun sa?",
            ],
            "field_questions_en": {
                "religion": "Do you follow a particular religion or spiritual practice? If yes, which one?",
                "mother_tongue": "What is your first language or mother tongue?",
            },
            "field_questions_hi": {
                "religion": "Kya aap koi dharm follow karte hain? Kaun sa?",
                "mother_tongue": "Aapki matra bhasha kya hai?",
            },
            "field_questions_hinglish": {
                "religion": "Aap koi religion follow karte ho? Kaun sa?",
                "mother_tongue": "Aapki mother tongue kya hai?",
            },
            "fields": ["religion", "mother_tongue"],
            "min_required": 1,
        },
        {
            "topic_id": "health_details",
            "topic_name": "Source of Referral",
            "natural_questions_en": [
                "How did you come to seek help today? Were you referred by a doctor, family member, friend, or self-referred?",
            ],
            "natural_questions_hi": [
                "Aaj aap madad lene kaise aaye? Kya kisi doctor, parivar ya dost ne refer kiya, ya aap khud aaye?",
            ],
            "natural_questions_hinglish": [
                "Aaj aap help lene kaise aaye? Doctor, family ya friend ne refer kiya, ya self-referred?",
            ],
            "fields": ["source_of_referral", "informant"],
            "min_required": 1,
        },
    ],
    "chief_complaints": [
        {
            "topic_id": "main_concern",
            "topic_name": "What Brings You Here",
            "natural_questions_en": [
                "What brings you here today?",
                "Can you describe what is bothering you the most right now?",
                "Are there any thoughts, feelings, or behaviours that are bothering you?",
                "What changes have you noticed in your mood, thinking, or behaviour recently?",
            ],
            "natural_questions_hi": [
                "Aap aaj yahan kis wajah se aaye hain?",
                "Aapko abhi sabse zyada kya pareshan kar raha hai?",
                "Koi khayal, ehsaas ya vyavhaar jo aapko pareshan kar raha hai?",
            ],
            "natural_questions_hinglish": [
                "Aap aaj yahan kyun aaye hain?",
                "Abhi sabse zyada kya bother kar raha hai aapko?",
                "Koi thoughts, feelings ya behaviour jo aapko pareshan kar rahe hain?",
            ],
            "fields": ["complaint_1"],
            "min_required": 1,
        },
        {
            "topic_id": "other_concerns",
            "topic_name": "Other Concerns",
            "natural_questions_en": [
                "Besides that, is there anything else that's been bothering you?",
            ],
            "natural_questions_hi": [
                "Iske alawa aur koi pareshani hai?",
            ],
            "natural_questions_hinglish": [
                "Iske alawa aur kuch bother kar raha hai?",
            ],
            "fields": ["complaint_2", "complaint_3"],
            "min_required": 0,
        },
        {
            "topic_id": "impact",
            "topic_name": "Duration & Impact",
            # Owns duration + daily-life impact. The hopi severity topic asks
            # only the 1-10 scale and trajectory — see question_overlaps.txt #4.
            "natural_questions_en": [
                "How long have you been experiencing these symptoms?",
            ],
            "natural_questions_hi": [
                "Ye lakshan aapko kab se ho rahe hain?",
            ],
            "natural_questions_hinglish": [
                "Ye symptoms kab se experience kar rahe ho?",
            ],
            "field_questions_en": {
                "duration_each": "How long have you been experiencing these symptoms?",
                "impact_on_functioning": "Are you having trouble at work, school, or in your personal relationships because of these issues?",
            },
            "field_questions_hi": {
                "duration_each": "Ye lakshan kab se hain?",
                "impact_on_functioning": "Kya inki wajah se kaam, padhai ya rishton mein dikkat ho rahi hai?",
            },
            "field_questions_hinglish": {
                "duration_each": "Ye symptoms kab se hain?",
                "impact_on_functioning": "Kya inki wajah se work, school ya relationships mein trouble ho rahi hai?",
            },
            "fields": ["duration_each", "impact_on_functioning"],
            "min_required": 1,
        },
    ],
    "factors_of_illness": [
        # Topic order and framing follow the docx: Predisposing -> Precipitating ->
        # Perpetuating -> Mode of onset -> Course -> Progress.
        {
            "topic_id": "background_triggers",
            "topic_name": "Recent Stresses & Triggers",
            # Docx "Precipitating factors" questions.
            "natural_questions_en": [
                "Can you think of any major stressors that occurred around the time your symptoms began?",
                "Did you experience any significant life changes or stress — like a job loss, relationship breakdown, or death of a loved one — that might have triggered these feelings?",
                "Have you been through any traumatic events, recently or in the past, that you feel might have contributed to your current emotional state?",
            ],
            "natural_questions_hi": [
                "Jab aapke lakshan shuru hue, us waqt koi bada stress ya tanav tha?",
                "Koi badi zindagi ki ghatna — naukri jaana, rishta tootna, kisi apne ka guzar jaana — jo iski wajah ban sakti hai?",
            ],
            "natural_questions_hinglish": [
                "Symptoms start hone ke time koi major stress tha?",
                "Koi badi life change ya stress — job loss, breakup, kisi ki death — jo trigger ban sakti hai?",
            ],
            "fields": ["predisposing_psychological", "precipitating_psychological"],
            "min_required": 1,
        },
        {
            "topic_id": "social_triggers",
            "topic_name": "Environment & Routine Changes",
            # Docx: "Has anything in your routine or environment changed recently...?"
            "natural_questions_en": [
                "Has anything in your routine or environment changed recently that could have triggered or worsened your symptoms — like moving homes, a change in your social circle, or a change in your work environment?",
            ],
            "natural_questions_hi": [
                "Kya haal mein aapke routine ya mahaul mein koi badlav aaya — ghar badalna, doston ka badalna, ya kaam ka mahaul?",
            ],
            "natural_questions_hinglish": [
                "Recently routine ya environment mein koi change aaya — ghar shift karna, social circle ya work environment ka change?",
            ],
            "fields": ["predisposing_social", "precipitating_social"],
            "min_required": 1,
        },
        {
            "topic_id": "health_triggers",
            "topic_name": "Family & Physical Health Factors",
            # Docx "Predisposing factors" (family history of mental illness) +
            # physical health question. The docx also asks family psychiatric
            # history again in family_history — that repeat is in the docx itself.
            "natural_questions_en": [
                "Is there a history of mental illness in your family?",
                "Have you experienced any serious physical health issues — like an illness, surgery, or injury — recently that might have affected your mental health?",
            ],
            "natural_questions_hi": [
                "Kya aapke parivar mein kisi ko mental illness rahi hai?",
                "Koi badi sharirik bimari, operation ya chot lagi thi jo aapke mann par asar kar sakti hai?",
            ],
            "natural_questions_hinglish": [
                "Family mein kisi ko mental illness ki history hai?",
                "Koi serious physical illness, surgery ya injury hui thi jo mental health ko affect kar sakti hai?",
            ],
            "fields": ["predisposing_biological", "precipitating_biological"],
            "min_required": 0,
        },
        {
            "topic_id": "keeping_it_going",
            "topic_name": "Ongoing Pressures",
            # Docx "Perpetuating factors" questions.
            "natural_questions_en": [
                "Are there any ongoing stressors or pressures in your life that are making it harder for you to manage your symptoms?",
                "Do you feel that work, family obligations, or personal relationships continue to cause significant stress?",
                "Do you have any habits — like alcohol or drug use, avoidance, or overworking — that might be preventing you from addressing the underlying issues?",
            ],
            "natural_questions_hi": [
                "Koi chal raha stress ya dabav hai jo aapke lakshan sambhalna mushkil bana raha hai?",
                "Kya kaam, parivar ki zimmedariyan ya rishte abhi bhi bahut stress de rahe hain?",
            ],
            "natural_questions_hinglish": [
                "Koi ongoing stress ya pressure hai jo symptoms manage karna mushkil bana raha hai?",
                "Kya work, family obligations ya relationships abhi bhi significant stress de rahe hain?",
            ],
            "fields": ["perpetuating_biological", "perpetuating_psychological", "perpetuating_social"],
            "min_required": 1,
        },
        {
            "topic_id": "illness_pattern",
            "topic_name": "How It Started & Changed",
            # Docx "Mode of onset", "Course of illness", "Progress of illness".
            "natural_questions_en": [
                "How did your symptoms first develop — were they sudden, or did they develop gradually over time?",
                "Did you notice a pattern in how your symptoms started?",
                "Were the symptoms more obvious in certain situations or times of the day — like at work, in social settings, or at night?",
            ],
            "natural_questions_hi": [
                "Aapke lakshan kaise shuru hue — achanak, ya dheere dheere samay ke saath?",
                "Kya aapne dhyan diya ki lakshan shuru hone ka koi pattern tha?",
                "Kya lakshan kuch khaas situations ya din ke kisi samay zyada dikhte the — jaise kaam par, logon ke beech, ya raat ko?",
            ],
            "natural_questions_hinglish": [
                "Symptoms kaise start hue — suddenly ya gradually over time?",
                "Kya aapne notice kiya ki symptoms start hone ka koi pattern tha?",
                "Kya symptoms kuch situations ya time of day mein zyada obvious the — jaise work par, social settings mein, ya raat ko?",
            ],
            "field_questions_en": {
                "mode_of_onset": "How did your symptoms first develop — were they sudden, or did they develop gradually over time?",
                "course_of_illness": "How have your symptoms changed since they first began?",
                "progress_of_illness": "Have your symptoms become stronger, more frequent, or harder to cope with as time has gone on?",
            },
            "field_questions_hi": {
                "mode_of_onset": "Lakshan achanak shuru hue ya dheere dheere?",
                "course_of_illness": "Shuruat se ab tak lakshan kaise badle hain?",
                "progress_of_illness": "Kya lakshan zyada tez, zyada baar, ya zyada pareshan karne wale ho gaye hain?",
            },
            "field_questions_hinglish": {
                "mode_of_onset": "Symptoms suddenly aaye ya gradually?",
                "course_of_illness": "Start hone ke baad se symptoms kaise change hue hain?",
                "progress_of_illness": "Kya symptoms zyada strong, zyada frequent ya handle karna zyada mushkil ho gaye hain?",
            },
            "fields": ["mode_of_onset", "course_of_illness", "progress_of_illness"],
            "min_required": 1,
        },
    ],
    "hopi": [
        {
            "topic_id": "symptom_details",
            "topic_name": "Symptom Details",
            "natural_questions_en": [
                "Can you describe exactly what you feel or experience?",
                "What does it feel like when this occurs?",
                "When it happens, what do you notice in your body or thoughts?",
            ],
            "natural_questions_hi": [
                "Aap exactly kya mahsoos karte hain? Kaisa hota hai jab ye hota hai?",
            ],
            "natural_questions_hinglish": [
                "Describe karo, exactly kya feel hota hai aapko?",
            ],
            "field_questions_en": {
                "character": "Can you describe exactly what you feel or experience?",
                "site": "Where do you feel it the most — in your body or your thoughts?",
                "onset_timing": "When did you first start experiencing this?",
            },
            "field_questions_hi": {
                "character": "Aap exactly kya mahsoos karte hain?",
                "site": "Sabse zyada kahan mahsoos hota hai — shareer mein ya vicharon mein?",
                "onset_timing": "Ye sabse pehle kab shuru hua tha?",
            },
            "field_questions_hinglish": {
                "character": "Exactly kya feel hota hai aapko?",
                "site": "Sabse zyada kahan feel hota hai — body mein ya thoughts mein?",
                "onset_timing": "Ye sabse pehle kab start hua tha?",
            },
            "fields": ["character", "site", "onset_timing"],
            "min_required": 1,
        },
        {
            "topic_id": "symptom_pattern",
            "topic_name": "Pattern & Triggers",
            "natural_questions_en": [
                "What makes it worse? And what helps you feel better?",
            ],
            "natural_questions_hi": [
                "Kya isko badhata hai? Aur kya isse aaram milta hai?",
            ],
            "natural_questions_hinglish": [
                "Kya isse worse banata hai? Aur kya help karta hai?",
            ],
            "fields": ["exacerbating_factors", "relieving_factors"],
            "min_required": 1,
        },
        {
            "topic_id": "severity",
            "topic_name": "Severity & Changes Over Time",
            "natural_questions_en": [
                "On a scale of 1 to 10, how severe is this right now?",
            ],
            "natural_questions_hi": [
                "1 se 10 ke scale par, abhi ye kitna severe hai?",
            ],
            "natural_questions_hinglish": [
                "1 se 10 ke scale par, abhi kitna severe hai?",
            ],
            "field_questions_en": {
                "severity": "On a scale of 1 to 10, how severe is this right now?",
                "time_course": "Have you ever experienced a period when your symptoms almost completely disappeared or significantly improved? And have they ever returned after a period of improvement?",
                "hospitalizations": "Can you tell me about the hospitalizations for this, if any?",
            },
            "field_questions_hi": {
                "severity": "1 se 10 ke scale par, abhi kitna severe hai?",
                "time_course": "Kya kabhi aisa hua ki lakshan lagbhag puri tarah gayab ho gaye ya bahut behtar ho gaye? Aur kya behtar hone ke baad wapas laute?",
                "hospitalizations": "Iske liye kabhi hospital mein admit hona pada?",
            },
            "field_questions_hinglish": {
                "severity": "1 se 10 mein rate karo, abhi kitna severe hai?",
                "time_course": "Kabhi aisa period aaya jab symptoms almost gayab ho gaye ya improve hue? Aur kya improvement ke baad wapas aaye?",
                "hospitalizations": "Iske liye kabhi hospitalization hui?",
            },
            "fields": ["severity", "time_course", "hospitalizations"],
            "min_required": 1,
        },
        {
            "topic_id": "serious_symptoms",
            "topic_name": "Thoughts of Harm",
            "natural_questions_en": [
                "I need to ask something important — have you had thoughts of harming yourself?",
                "Have you ever felt like life isn’t worth living?",
                "Have there been moments where you wished you could just disappear?",
                "Have you had thoughts of ending your life?",
                "Sometimes when people feel this way, they have thoughts of hurting themselves — has that happened to you?",
                "Have you ever felt like you don’t want to go on?",
                "Have you had any thoughts about self-harm recently?",
                "Have things ever felt so overwhelming that you thought about harming yourself?",
                "Have you experienced thoughts about not wanting to be here anymore?",
                "Have you had any urges to hurt yourself or end your life?",
                "Have you made any plans to harm yourself?",
                "Have you ever acted on these thoughts before?",
                "Do these thoughts come and go, or are they persistent?",
                "When these thoughts come, how strong do they feel?",
                "Have you been able to stay safe when these thoughts occur?"
            ],
            "natural_questions_hi": [
                "Kya aapne apne aap ko nuqsan pahunchane ke baare mein soca hai?",
            ],
            "natural_questions_hinglish": [
                "Have you thought about hurting yourself?",
            ],
            "fields": ["suicidality_assessed"],
            "min_required": 1,
        },
    ],

    # NEGATIVE HISTORY 
    "negative_history": [
        {
            "topic_id": "cardio_respiratory",
            "topic_name": "Heart & Breathing",
            "natural_questions_en": [
                "Have you had any chest pain, palpitations, or breathlessness?",
                "Any issues with your heart or breathing that you've noticed?",
            ],
            "natural_questions_hi": [
                "Kya aapko seene mein dard, dhadkan ki takleef ya saans lene mein pareshani hui hai?",
            ],
            "natural_questions_hinglish": [
                "Chest pain, palpitations, ya breathlessness — koi aisa hua hai?",
            ],
            "field_questions_en": {
                "cvs_negative": "Any chest pain or palpitations?",
                "respiratory_negative": "Any breathing problems or persistent cough?",
            },
            "field_questions_hi": {
                "cvs_negative": "Seene mein dard ya dhadkan ki takleef?",
                "respiratory_negative": "Saans lene mein koi takleef?",
            },
            "field_questions_hinglish": {
                "cvs_negative": "Chest pain ya heart racing?",
                "respiratory_negative": "Breathlessness ya cough?",
            },
            "fields": ["cvs_negative", "respiratory_negative"],
            "min_required": 1,
        },
        {
            "topic_id": "gastro_neuro",
            "topic_name": "Digestion & Nervous System",
            "natural_questions_en": [
                "Have you had any stomach problems — nausea, changes in appetite, or digestion issues? Any headaches, fits, or numbness?",
                "Any issues with your stomach or gut? Or headaches, fits, memory problems?",
            ],
            "natural_questions_hi": [
                "Pet mein koi takleef? Ya sar dard, fits, ya numbness?",
            ],
            "natural_questions_hinglish": [
                "Stomach issues ya headaches, fits, numbness — kuch aisa?",
            ],
            "field_questions_en": {
                "gi_negative": "Any stomach or digestive issues?",
                "neuro_negative": "Any headaches, fits, or numbness?",
            },
            "field_questions_hi": {
                "gi_negative": "Pet ya pachhan mein koi takleef?",
                "neuro_negative": "Sar dard, fits ya numbness?",
            },
            "field_questions_hinglish": {
                "gi_negative": "Stomach ya digestion mein kuch?",
                "neuro_negative": "Headaches, fits, ya numbness?",
            },
            "fields": ["gi_negative", "neuro_negative"],
            "min_required": 1,
        },
        {
            "topic_id": "urinary_musculoskeletal",
            "topic_name": "Urinary & Joints",
            "natural_questions_en": [
                "Any problems with urination or your urinary system? Any joint or muscle pain?",
            ],
            "natural_questions_hi": [
                "Peshab mein koi takleef? Ya jodon ya muscles mein dard?",
            ],
            "natural_questions_hinglish": [
                "Urinary problems ya joint/muscle pain?",
            ],
            "fields": ["genitourinary_negative", "musculoskeletal_negative"],
            "min_required": 1,
        },
        {
            "topic_id": "endocrine_screening",
            "topic_name": "Hormonal & Metabolic",
            "natural_questions_en": [
                "Any symptoms suggesting thyroid problems — like weight changes, feeling very hot or cold, or excessive fatigue? Any history of diabetes?",
                "Any hormonal or metabolic issues you know of?",
            ],
            "natural_questions_hi": [
                "Thyroid ya diabetes ke koi lakshan? Wazn badalna ya bohot thakaan?",
            ],
            "natural_questions_hinglish": [
                "Thyroid ya diabetes — koi symptoms?",
            ],
            "fields": ["endocrine_negative"],
            "min_required": 1,
        },
    ],

    # PAST HISTORY
    "past_history": [
        {
            "topic_id": "past_psychiatric_episodes",
            "topic_name": "Previous Mental Health Episodes",
            "natural_questions_en": [
                "Have you ever gone through something like this before — an earlier time when your mental or emotional health was a real struggle?",
                "Can you describe what you went through back then and when it first started?",
            ],
            "natural_questions_hi": [
                "Kya pehle kabhi aisa daur aaya tha jab aapke mann ya bhaavnaon se judi pareshani rahi ho?",
            ],
            "natural_questions_hinglish": [
                "Kya pehle kabhi aisa time aaya tha jab aapki mental ya emotional health kaafi mushkil mein thi?",
            ],
            "field_questions_en": {
                "previous_psychiatric_episodes": "Can you describe the symptoms you experienced and when they first occurred?",
                "previous_diagnoses": "What were you diagnosed with?",
            },
            "field_questions_hi": {
                "previous_psychiatric_episodes": "Us waqt kya lakshan the aur kab shuru hue the?",
                "previous_diagnoses": "Aapko kya diagnosis bataya gaya tha?",
            },
            "field_questions_hinglish": {
                "previous_psychiatric_episodes": "Us time kya symptoms the aur kab start hue the?",
                "previous_diagnoses": "Aapko kya diagnose kiya gaya tha?",
            },
            "fields": ["previous_psychiatric_episodes", "previous_diagnoses"],
            "min_required": 1,
        },
        {
            "topic_id": "past_treatment",
            "topic_name": "Treatment History",
            "natural_questions_en": [
                "What kind of treatments did you receive? Did this treatment help with your symptoms?",
            ],
            "natural_questions_hi": [
                "Aapko kis tarah ka ilaaj mila? Kya us ilaaj se lakshanon mein fayda hua?",
            ],
            "natural_questions_hinglish": [
                "Aapko kya treatments mile? Kya unse symptoms mein help hui?",
            ],
            "field_questions_en": {
                "treatment_history": "What kind of treatments did you receive, and did they help with your symptoms?",
                "medications_past": "What all medications were prescribed to you before?",
                "hospitalizations_past": "Can you tell me about the hospitalizations, if any — like reasons for admission, duration of stay, and treatment outcomes?",
            },
            "field_questions_hi": {
                "treatment_history": "Kis tarah ka ilaaj mila aur kya fayda hua?",
                "medications_past": "Pehle aapko kaun kaun si dawaiyan di gayi thin?",
                "hospitalizations_past": "Kabhi hospital mein admit hue the? Kis wajah se, kitne din, aur kya natija raha?",
            },
            "field_questions_hinglish": {
                "treatment_history": "Kaise treatments mile aur kya help hui?",
                "medications_past": "Pehle kaun si medications prescribe hui thin?",
                "hospitalizations_past": "Koi hospitalizations hui thin? Reason, duration aur outcome kya tha?",
            },
            "fields": ["treatment_history", "medications_past", "hospitalizations_past"],
            "min_required": 1,
        },
        {
            "topic_id": "past_medical_surgical",
            "topic_name": "Medical & Surgical History",
            "natural_questions_en": [
                "Have you had any major medical illnesses, surgeries, or conditions that required regular treatment or hospitalisation?",
                "Any significant physical health problems in the past — chronic illnesses, operations, or emergency hospital visits?",
            ],
            "natural_questions_hi": [
                "Koi badi bimari, operation, ya hospital mein admit rehna pada?",
            ],
            "natural_questions_hinglish": [
                "Koi major illness, surgery ya hospital admission hua tha physically?",
            ],
            "fields": ["medical_surgical_history", "drug_history_current"],
            "min_required": 1,
        },
        {
            "topic_id": "allergies",
            "topic_name": "Allergies",
            "natural_questions_en": [
                "Do you have any known allergies — to medications, foods, or anything else?",
            ],
            "natural_questions_hi": [
                "Kisi cheez se allergy hai — dawaiyan, khaana ya kuch aur?",
            ],
            "natural_questions_hinglish": [
                "Koi allergies — medicines, food, ya kuch aur?",
            ],
            "fields": ["allergies"],
            "min_required": 1,
        },
    ],

    # FAMILY HISTORY 
    "family_history": [
        {
            "topic_id": "family_structure",
            "topic_name": "Family Structure",
            "natural_questions_en": [
                "Who all are there in your family?",
                "Do you live alone or with your family?",
            ],
            "natural_questions_hi": [
                "Aapke parivar mein kaun kaun hain?",
            ],
            "natural_questions_hinglish": [
                "Aapki family mein kaun kaun hai?",
            ],
            "field_questions_en": {
                "family_members_list": "Who all are there in your family?",
                "living_arrangement": "Do you live alone or with your family?",
            },
            "field_questions_hi": {
                "family_members_list": "Parivar mein kaun kaun hain?",
                "living_arrangement": "Aap akele rehte hain ya parivar ke saath?",
            },
            "field_questions_hinglish": {
                "family_members_list": "Family mein kaun kaun hai?",
                "living_arrangement": "Aap akele rehte ho ya family ke saath?",
            },
            "fields": ["family_members_list", "living_arrangement"],
            "min_required": 2,
        },
        {
            "topic_id": "family_members_background",
            "topic_name": "Family Members' Background",
            "natural_questions_en": [
                "What is your family members' highest level of education, and what are their occupations?",
            ],
            "natural_questions_hi": [
                "Parivar ke logon ki padhai kahan tak hui hai? Aur woh kya kaam karte hain?",
            ],
            "natural_questions_hinglish": [
                "Family members ki education kahan tak hui hai? Aur unka occupation kya hai?",
            ],
            "fields": ["education_occupation_members"],
            "min_required": 1,
        },
        {
            "topic_id": "family_leadership",
            "topic_name": "Family Leadership",
            "natural_questions_en": [
                "In your family, who takes care of running things day to day? And who is usually seen as the head of the family?",
            ],
            "natural_questions_hi": [
                "Aapke parivar mein ghar ke rozmarra ke kaam asal mein kaun sambhalta hai? Aur mukhiya kaun mana jaata hai?",
            ],
            "natural_questions_hinglish": [
                "Aapke family mein ghar rozana actually kaun chalata hai? Aur head kaun mana jaata hai?",
            ],
            "field_questions_en": {
                "functional_head": "Who actually takes care of running your household day to day?",
                "nominal_head": "And who is usually seen as the head of the family?",
            },
            "field_questions_hi": {
                "functional_head": "Ghar ke rozmarra ke kaam asal mein kaun sambhalta hai?",
                "nominal_head": "Parivar ka mukhiya kaun mana jaata hai?",
            },
            "field_questions_hinglish": {
                "functional_head": "Ghar rozana actually kaun chalata hai?",
                "nominal_head": "Family ka head kaun mana jaata hai?",
            },
            "fields": ["functional_head", "nominal_head"],
            "min_required": 1,
        },
        {
            "topic_id": "family_relationships",
            "topic_name": "Family Relationships",
            "natural_questions_en": [
                "Whom are you closest with in your family?",
                "How would you describe your relationship with your parents or siblings? Do you have a close-knit family?",
            ],
            "natural_questions_hi": [
                "Parivar mein aap kiske sabse kareeb hain?",
            ],
            "natural_questions_hinglish": [
                "Family mein aap kiske sabse close ho?",
            ],
            "field_questions_en": {
                "relationship_quality": "How would you describe your relationship with your parents or siblings?",
                "family_dynamics": "How do family members typically interact with one another — are there any patterns of conflict, avoidance, or support?",
                "support_system": "Do you feel supported by your family? Are there certain family members you turn to for emotional support?",
            },
            "field_questions_hi": {
                "relationship_quality": "Mata-pita ya bhai-behnon ke saath aapka rishta kaisa hai?",
                "family_dynamics": "Parivar ke log aapas mein kaise rehte hain — koi takraav, doori ya sahara?",
                "support_system": "Kya aapko parivar ka sahara mehsoos hota hai? Kis se emotional support milta hai?",
            },
            "field_questions_hinglish": {
                "relationship_quality": "Parents ya siblings ke saath relationship kaisa hai?",
                "family_dynamics": "Family members aapas mein kaise interact karte hain — koi conflict, avoidance ya support?",
                "support_system": "Kya family se supported feel karte ho? Emotional support ke liye kiske paas jaate ho?",
            },
            "fields": ["relationship_quality", "family_dynamics", "support_system"],
            "min_required": 1,
        },
        {
            "topic_id": "family_illness_history",
            "topic_name": "Family Health History",
            "natural_questions_en": [
                "Is there any history of psychiatric illness in your family?",
            ],
            "natural_questions_hi": [
                "Kya aapke parivar mein kisi ko psychiatric bimari rahi hai?",
            ],
            "natural_questions_hinglish": [
                "Family mein kisi ko psychiatric illness ki history hai?",
            ],
            "field_questions_en": {
                "psychiatric_illness_in_family": "Has any family member had a mental health condition or been treated by a psychiatrist?",
                "genetic_conditions": "Any genetic or hereditary conditions in the family?",
            },
            "field_questions_hi": {
                "psychiatric_illness_in_family": "Parivar mein kisi ko mental illness?",
                "genetic_conditions": "Koi genetic ya hereditary bimari?",
            },
            "field_questions_hinglish": {
                "psychiatric_illness_in_family": "Family mein kisi ko mental illness thi?",
                "genetic_conditions": "Koi genetic condition family mein?",
            },
            "fields": ["psychiatric_illness_in_family", "genetic_conditions"],
            "min_required": 1,
        },
    ],

    # PERSONAL HISTORY
    "personal_history": [
        {
            "topic_id": "birth_development",
            "topic_name": "Birth & Early Development",
            # These are informant-level details the patient often won't know
            # first-hand, so they're framed as "from what you've been told" and
            # it's fine if the patient isn't sure.
            "natural_questions_en": [
                "This next one you might only know from what family has told you — as far as you're aware, was your birth and early childhood normal, or were there any complications? It's completely fine if you're not sure.",
            ],
            "natural_questions_hi": [
                "Ye baat shayad aapko sirf ghar walon se pata ho — jitna aapko maloom hai, kya aapka janm aur bachpan theek tha, ya koi dikkat thi? Agar pakka nahi pata to koi baat nahi.",
            ],
            "natural_questions_hinglish": [
                "Ye shayad aapko sirf family se pata ho — jitna aapko pata hai, kya aapka birth aur early childhood normal tha, ya koi complications thi? Agar sure nahi ho to bilkul koi baat nahi.",
            ],
            "field_questions_en": {
                "birth_and_early_development": "From what you've been told, were there any complications around your birth or health concerns soon after? (No worries if you don't know.)",
                "developmental_milestones": "As far as your family has mentioned, did you reach the usual milestones — walking, talking, starting school — around the expected time?",
            },
            "field_questions_hi": {
                "birth_and_early_development": "Jitna aapko bataya gaya hai, kya janm ke samay ya turant baad koi dikkat thi? (Na pata ho to koi baat nahi.)",
                "developmental_milestones": "Jitna ghar walon ne bataya ho — chalna, bolna, school jaana samay par hua tha?",
            },
            "field_questions_hinglish": {
                "birth_and_early_development": "Jitna aapko bataya gaya hai, kya birth ke time ya turant baad koi complications thi? (Na pata ho to koi baat nahi.)",
                "developmental_milestones": "Jitna family ne bataya ho — walking, talking, school samay par shuru hua tha?",
            },
            "fields": ["birth_and_early_development", "developmental_milestones"],
            "min_required": 1,
        },
        {
            "topic_id": "childhood_school",
            "topic_name": "Childhood & School",
            "natural_questions_en": [
                "Growing up, how would you describe your relationship with your parents or the people who raised you?",
                "How did you perform in school as a child and adolescent?",
            ],
            "natural_questions_hi": [
                "Bachpan mein aapka apne maata-pita ya dekhbhal karne walon ke saath rishta kaisa tha?",
            ],
            "natural_questions_hinglish": [
                "Bachpan mein aapka apne parents ya jinhone aapko paala, unke saath relationship kaisa tha?",
            ],
            "field_questions_en": {
                "childhood_environment": "Were you able to talk to your parents about your feelings, concerns, or problems, or did you tend to keep things to yourself?",
                "school_history": "How did you perform in school as a child and adolescent — were you an average student, or did you excel or struggle in certain subjects?",
                "peer_relationships_childhood": "How did you interact with other children during your early years — were you able to make friends easily?",
            },
            "field_questions_hi": {
                "childhood_environment": "Kya aap apne maata-pita se apni feelings ya pareshaniyan share kar paate the, ya cheezein apne tak rakhte the?",
                "school_history": "School mein padhai kaisi thi — average the, ya kuch subjects mein bahut acche ya kamzor?",
                "peer_relationships_childhood": "Bachpan mein doosre bachchon ke saath kaise the — dost asaani se ban jaate the?",
            },
            "field_questions_hinglish": {
                "childhood_environment": "Kya aap parents se feelings share kar paate the, ya cheezein khud tak rakhte the?",
                "school_history": "School mein performance kaisi thi — average, excel ya struggle?",
                "peer_relationships_childhood": "Bachpan mein doosre bachchon ke saath kaise the — friends easily bante the?",
            },
            "fields": ["childhood_environment", "school_history", "peer_relationships_childhood"],
            "min_required": 1,
        },
        {
            "topic_id": "occupational_history",
            "topic_name": "Occupational History",
            "natural_questions_en": [
                "Are you currently employed? If yes, what is your current job title or role?",
                "What jobs have you had before your current role? How long did you stay in each, and why did you leave?",
            ],
            "natural_questions_hi": [
                "Kya aap abhi kaam kar rahe hain? Aapka abhi ka pad kya hai?",
                "Isse pehle aapne kaun kaun si naukriyan ki hain? Kitne samay tak, aur kyun chhodi?",
            ],
            "natural_questions_hinglish": [
                "Kya aap currently employed ho? Current job title kya hai?",
                "Current role se pehle kaun si jobs ki hain? Kitna time, aur kyun chhodi?",
            ],
            "fields": ["occupational_history"],
            "min_required": 1,
        },
        {
            "topic_id": "relationships_reproductive",
            "topic_name": "Marital & Sexual History",
            "natural_questions_en": [
                "Are you currently married? If so, how long have you been together? If not, have you ever been married?",
            ],
            "natural_questions_hi": [
                "Kya aapki shaadi hui hai? Agar haan, to kitne samay se saath hain? Agar nahi, to kya pehle kabhi hui thi?",
            ],
            "natural_questions_hinglish": [
                "Kya aap currently married ho? Kitne time se saath ho? Ya pehle kabhi marriage hui thi?",
            ],
            "field_questions_en": {
                "marital_history": "How would you describe the quality of your current or most significant relationship?",
                "reproductive_history": "Do you have children? If so, how many and what are their ages?",
                "sexual_development": "How did you experience puberty in terms of your development?",
            },
            "field_questions_hi": {
                "marital_history": "Aapke sabse khaas rishtey ki quality kaisi hai ya thi?",
                "reproductive_history": "Kya aapke bachche hain? Kitne, aur unki umar kya hai?",
                "sexual_development": "Puberty ke samay aapka anubhav kaisa raha?",
            },
            "field_questions_hinglish": {
                "marital_history": "Aapke current ya most significant relationship ki quality kaisi hai?",
                "reproductive_history": "Bachche hain? Kitne aur kya age hai?",
                "sexual_development": "Puberty ka experience kaisa raha development ke terms mein?",
            },
            "fields": ["marital_history", "reproductive_history", "sexual_development"],
            "min_required": 1,
        },
        {
            "topic_id": "substance_use",
            "topic_name": "Substance Use",
            "natural_questions_en": [
                "Have you ever used alcohol, tobacco, or any illicit substances? If yes, what substances and how often?",
            ],
            "natural_questions_hi": [
                "Kya aapne kabhi sharaab, tambaaku ya koi nasha kiya hai? Agar haan, to kya aur kitni baar?",
            ],
            "natural_questions_hinglish": [
                "Kabhi alcohol, tobacco ya koi substances use kiye hain? Kya aur kitni often?",
            ],
            "fields": ["substance_use_history"],
            "min_required": 1,
        },
        {
            "topic_id": "premorbid_personality",
            "topic_name": "Personality Before This Started",
            "natural_questions_en": [
                "Before all this started, how would you describe your social life?",
                "Before all this started, how would you describe your usual mood or emotional state?",
                "How did you typically respond when faced with stress or difficult situations before your current difficulties?",
            ],
            "natural_questions_hi": [
                "Apni abhi ki pareshaniyon se pehle aapki social life kaisi thi?",
                "In dikkaton se pehle aapka aam mood ya bhavnatmak sthiti kaisi thi?",
            ],
            "natural_questions_hinglish": [
                "Current issues se pehle aapki social life kaisi thi?",
                "In difficulties se pehle aapka general mood kaisa rehta tha?",
            ],
            "fields": ["premorbid_personality", "forensic_history"],
            "min_required": 1,
        },
    ],

    "mse": [
        {
            "topic_id": "mood_affect",
            "topic_name": "Mood & Emotions",
            "natural_questions_en": [
                "How would you describe your mood lately?",
            ],
            "natural_questions_hi": [
                "Aaj kal aap apna mood kaisa batayenge?",
            ],
            "natural_questions_hinglish": [
                "Aaj kal aapka mood kaisa hai?",
            ],
            "field_questions_en": {
                "mood_subjective": "How would you describe your mood in your own words?",
                "mood_objective": "How would others describe your mood from the outside?",
                "affect_quality": "What emotion is most present for you right now?",
                "affect_intensity": "How intense are your emotions — very flat, or very strong?",
                "affect_mobility": "Does your mood change a lot throughout the day, or does it stay mostly the same?",
            },
            "field_questions_hi": {
                "mood_subjective": "Aap apna mood kaisa batayenge?",
                "affect_intensity": "Emotions kitni strong hain?",
                "affect_mobility": "Mood din mein badalta rehta hai ya same rehta hai?",
            },
            "field_questions_hinglish": {
                "mood_subjective": "Mood ko apne words mein describe karo?",
                "affect_intensity": "Emotions kitni strong hain?",
                "affect_mobility": "Mood stable hai ya bahut change hota hai?",
            },
            "fields": ["mood_subjective", "mood_objective", "affect_quality", "affect_intensity", "affect_mobility"],
            "min_required": 2,
        },
        {
            "topic_id": "thought_content",
            "topic_name": "Thoughts",
            "natural_questions_en": [
                "Are there any thoughts that keep coming back to you — worries, fears, or anything you can't seem to stop thinking about?",
                "What kinds of thoughts are going through your mind most often these days?",
            ],
            "natural_questions_hi": [
                "Koi baar baar aane wale khayal hain? Koi darr, chinta ya aisi soch jo band nahi hoti?",
            ],
            "natural_questions_hinglish": [
                "Koi thoughts jo baar baar aate hain — worries, fears?",
            ],
            "field_questions_en": {
                "thought_content": "Are there specific worries or ideas that occupy your mind a lot?",
                "thought_stream": "Do your thoughts feel slow, fast, or normal?",
                "thought_form": "Do you find it easy to follow your own train of thought?",
                "thought_possession": "Have you ever felt like your thoughts were not your own, or being put into your mind?",
            },
            "field_questions_hi": {
                "thought_content": "Koi khaas khayal jo baar baar aate hain?",
                "thought_stream": "Soch dheemi lagti hai ya bahut tez?",
                "thought_possession": "Kabhi laga ki khayal apne nahi hain?",
            },
            "field_questions_hinglish": {
                "thought_content": "Koi specific thoughts jo baar baar aate hain?",
                "thought_stream": "Soch slow lagti hai ya racing?",
            },
            "fields": ["thought_stream", "thought_form", "thought_content", "thought_possession"],
            "min_required": 1,
        },
        {
            "topic_id": "perception",
            "topic_name": "Unusual Experiences",
            "natural_questions_en": [
                "Have you had any unusual experiences — like hearing voices when no one is there, or seeing things others don't see?",
                "Sometimes people going through difficult times have unusual perceptual experiences. Has anything like that happened to you?",
            ],
            "natural_questions_hi": [
                "Kya kabhi aawazein sunai deen jab koi nahi tha? Ya kuch aisa dikh jata hai jo doosron ko nahi?",
            ],
            "natural_questions_hinglish": [
                "Koi unusual experience — voices sunna ya kuch aisa dekhna jo doosre nahi dekhte?",
            ],
            "fields": ["perception"],
            "min_required": 1,
        },
        {
            "topic_id": "cognition",
            "topic_name": "Attention & Orientation",
            "natural_questions_en": [
                "I'd like to do a quick attention exercise. Please start from 100 and subtract 7 each time — give me the first five numbers.",
            ],
            "natural_questions_hi": [
                "Ek chhota sa dhyan ka abhyas karte hain. 100 se shuru karke har baar 7 ghataiye — pehle paanch ank bataiye.",
            ],
            "natural_questions_hinglish": [
                "Ek quick attention exercise karte hain. 100 se start karke har baar 7 minus karo — pehle paanch numbers batao.",
            ],
            "field_questions_en": {
                "attention_concentration": "Please count backwards starting from 20, one number at a time.",
                "orientation": "Can you tell me today's date, the day of the week, and where you are right now?",
                "memory": "Have you had any trouble remembering things recently?",
            },
            "field_questions_hi": {
                "attention_concentration": "Kripya 20 se ulti ginti kijiye, ek ek karke.",
                "orientation": "Aaj ki tareekh, din, aur aap abhi kahan hain — bata sakte hain?",
                "memory": "Haal mein kuch yaad rakhne mein dikkat hui hai?",
            },
            "field_questions_hinglish": {
                "attention_concentration": "20 se backwards count karo, ek ek karke.",
                "orientation": "Aaj ki date, day, aur aap abhi kahan ho — bata sakte ho?",
                "memory": "Recently kuch yaad rakhne mein problem hui hai?",
            },
            "fields": ["consciousness", "attention_concentration", "orientation", "memory", "general_knowledge"],
            "min_required": 1,
        },
        {
            "topic_id": "judgment_insight",
            "topic_name": "Self-Awareness & Judgment",
            "natural_questions_en": [
                "Do you have a sense of what might be going on with you — do you feel like something is wrong and that you need help?",
                "How do you understand what you're going through? Do you feel it's an illness, or something else?",
            ],
            "natural_questions_hi": [
                "Aapko lagta hai kya ki koi bimari hai? Kya aapko lagta hai madad ki zaroorat hai?",
            ],
            "natural_questions_hinglish": [
                "Aapko kya lagta hai — koi illness hai? Help chahiye?",
            ],
            "field_questions_en": {
                "insight": "Do you feel that what you're experiencing is an illness that needs treatment?",
                "judgment": "If you lost your wallet with all your important documents inside, what would you do?",
            },
            "field_questions_hi": {
                "insight": "Aapko lagta hai ye bimari hai aur ilaaj chahiye?",
                "judgment": "Agar aapka wallet kho jaaye jismein aapke zaroori documents hon, to aap kya karenge?",
            },
            "field_questions_hinglish": {
                "insight": "Lagta hai ye illness hai aur treatment chahiye?",
                "judgment": "Agar aapka wallet kho jaaye jisme important documents hain, to aap kya karoge?",
            },
            "fields": ["judgment", "insight"],
            "min_required": 1,
        },
        {
            "topic_id": "appearance_speech",
            "topic_name": "Current State",
            "natural_questions_en": [
                "How have you been taking care of yourself lately — sleep, eating, personal hygiene?",
                "How would you describe your energy and how you've been presenting yourself day to day?",
            ],
            "natural_questions_hi": [
                "Aaj kal apna khayal kaise rakh rahe ho — neend, khaana, saaf-safai?",
            ],
            "natural_questions_hinglish": [
                "Self-care kaisa hai — sleep, eating, hygiene?",
            ],
            "fields": ["general_appearance_behaviour", "psychomotor_activity", "speech_characteristics"],
            "min_required": 1,
        },
        {
            "topic_id": "higher_cognition",
            "topic_name": "General Knowledge & Reasoning",
            "natural_questions_en": [
                "I'd like to ask a few brief questions to get a picture of how your thinking is — is that okay?",
            ],
            "natural_questions_hi": [
                "Kuch chhote sawaal poochne hain sochne ki kshamata ke baare mein — theek hai?",
            ],
            "natural_questions_hinglish": [
                "Kuch brief questions — thinking aur reasoning ke baare mein?",
            ],
            "fields": ["intelligence", "abstract_ability", "general_knowledge"],
            "min_required": 1,
        },
    ],
}

FIELD_HUMANIZE = {
    "en": {
        "name": "name", "age": "age", "sex": "gender", "education": "education",
        "occupation": "occupation", "ses": "socioeconomic background",
        "marital_status": "marital status", "religion": "religion",
        "residence": "residence", "mother_tongue": "mother tongue",
        "address": "address", "source_of_referral": "referral source",
        "informant": "informant", "family_type": "family type",
        "complaint_1": "main concern", "complaint_2": "second concern",
        "complaint_3": "third concern", "duration_each": "duration",
        "impact_on_functioning": "daily impact",
    },
    "hi": {
        "name": "naam", "age": "umra", "sex": "ling", "education": "shiksha",
        "occupation": "kaam", "ses": "aarthik sthiti",
        "marital_status": "vaivaahik sthiti", "religion": "dharm",
        "residence": "niwas", "mother_tongue": "maatri bhaasha",
        "address": "pata", "source_of_referral": "referral source",
        "informant": "soochna daata", "family_type": "parivar ka prakar",
        "complaint_1": "mukhya shikayat", "duration_each": "avdhi",
        "impact_on_functioning": "prabhav",
    },
    "hinglish": {
        "name": "naam", "age": "age", "sex": "gender", "education": "education",
        "occupation": "kaam", "ses": "financial situation",
        "marital_status": "relationship status", "religion": "dharm",
        "residence": "residence", "mother_tongue": "mother tongue",
        "address": "address", "source_of_referral": "referral source",
        "informant": "informant", "family_type": "family type",
        "complaint_1": "main problem", "duration_each": "kitne time se",
        "impact_on_functioning": "daily life impact",
    }
}