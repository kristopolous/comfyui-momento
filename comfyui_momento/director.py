"""Prompts, kept free of ComfyUI imports so scripts can read them."""

# What Gemini is told. It lives in the graph (Gemini's system prompt widget); this copy is the default.
DIRECTOR = """Left image: someone's profile picture. Right image: a photo they posted.
Describe the real moment that photo was taken, as a NEW camera angle on the same place: {angle} Everything the post
shows (the subject, the buildings, the objects, the light, the weather, the time of day) is still there, now seen from
this new position, along with the surroundings the post cropped out. The photographer is the person in the profile
picture (if it doesn't show a person, an anthropomorphic rabbit in everyday clothes), holding a phone up, aimed at the
subject of the post. Keep the person exactly as the profile picture shows them: same face, hair and build; don't add
jewelry, earrings, glasses, hats or accessories the profile picture doesn't show. Realistic and flattering. The subject of the post is
active, never standing there posing: the photographer catches a MOMENT in time, the instant the post froze. No jokes, no mishaps, no dialogue. Reply with JSON only:
{{"still": "<instruction for an image generator: one vertical frame of exactly that view. Name every element of the
   post and where it sits in this new view, then the photographer and the phone. The person is <person_1> from
   ref_image_1. {still}>",
 "boxes": [<FLUX 3 layout for the still, boxes as [y_min, x_min, y_max, x_max] on a 0-1000 grid of each image:
   {{"id": "photographer", "from": "ref_image_1", "src_bbox": <where the person is in the profile picture>,
    "tgt_bbox": <where the photographer is in the new frame>, "desc": "<the photographer, from behind or side>"}},
   plus a row for the phone and for each main thing from the post you place, "from": "ref_image_0" with its
   "src_bbox" in the post. Cite each id in "still" as <id>.>],
 "before": "<instruction for an image editor, given that frame as ref_image_0 and the person as ref_image_1: the
   same frame about one second EARLIER. Same place, light and camera; the photographer the same, phone already up. The
   subject of the post is mid-action, a beat before the instant the post froze: name what they are doing (a singer
   lunging toward the mic, a skater crouched before the jump, a crowd before the confetti). Not posed, not still.>",
 "video": "<prompt for an image-to-video model, 4 seconds, no speech: {motion} Sound: the ambient sound of the place,
   and a crisp camera lens shutter sound at the moment the photo is taken. Nobody talks.>",
 "music": "<a prompt for an instrumental music generator that fits THIS post's place, time and mood, from what you see:
   genre, instruments, tempo and feeling in one or two sentences (a punk show at night gets fast distorted punk; a
   quiet park at dawn gets something gentle; a city street gets the sound of that city).>"}}"""

# camera positions: the view Gemini plans, what happens over the 4 seconds, and how the still relates to the post
NEW_VIEW = "This is a different camera position from ref_image_0, not a copy."
ANGLES = {
    "wide": ("a bystander's view from a few steps directly behind the photographer: the post's exact view is in the "
             "middle of the frame, with the surroundings the post cropped out all around it (the rest of the place, "
             "above, below and to the sides), and the photographer's head, shoulders and raised phone in the "
             "foreground at the bottom, seen from behind, the phone aimed at the scene.",
             "the subject of the post is in motion, doing what they do (name it), and moves into the exact instant the "
             "post froze; the photographer, phone up, catches it: the shutter is tapped on that instant. Steady "
             "handheld camera from behind the photographer, a slight push in. Everything around keeps moving.",
             "The middle of the frame shows the post's scene, the subject in exactly the post's pose (layout id <scene>, placed "
             "for you), part of the real place, never a picture, poster or screen: describe the wider place around it, extended "
             "seamlessly in the same light, lens and style, and the photographer seen from behind in the foreground."),
    "behind": ("a bystander's view from behind and to the side of the photographer, a few meters back, with the phone's "
               "screen visible showing the shot being framed.",
               "the photographer steps into position, raises the phone and frames the shot while the handheld camera "
               "moves in over their shoulder, ending on exactly what the phone sees, the moment the shutter is "
               "tapped.", NEW_VIEW),
    "face": ("a bystander's view from the side and slightly in front of the photographer, about two meters away, so "
             "their face is clearly visible in three-quarter view, concentrating, phone raised toward the subject, "
             "which is visible beside or behind them.",
             "the photographer raises the phone, frames the shot with a small concentrated look, taps the shutter and "
             "lowers the phone with a quiet smile. Handheld bystander camera, gently drifting.", NEW_VIEW),
}


def director(angle="behind", camera="smartphone"):
    """camera: what the photographer holds, "smartphone" or "dslr"."""
    view, motion, still = ANGLES[angle]
    text = DIRECTOR.format(angle=view, motion=motion, still=still)
    if camera == "dslr":
        for a, b in (("holding a phone up", "holding a DSLR camera with a lens up to their eye"),
                     ("taps the shutter", "presses the shutter"), ("tapped", "pressed"),
                     ("phone camera shutter", "DSLR shutter"), ("the phone sees", "the camera sees"),
                     ("phone's screen visible showing", "camera raised, its lens aimed at"), ("phone", "camera")):
            text = text.replace(a, b)
    return text


# The critic: Gemini watches each take and rejects bad ones, which get reshot with a new seed.
CRITIC = """You are the editor of an app that films the moment someone took a photo. Image, left: the
photographer's profile picture. Image, right: the real photo they posted. Video: a generated take of the moment that
photo was taken, ending on that photo. It is a small, low-resolution preview: some softness in faces, hands and
small details is expected and fine. Reject the take only for failures a viewer would notice at a glance:
- the photographer doesn't look like the profile picture's person (or, if that isn't a person, isn't an
  anthropomorphic rabbit), or turns into someone or something else during the clip
- faces, hands, the phone or the camera visibly melt, multiply or morph into something else
- something pushes into or covers the lens
- the subject of the photo just stands there instead of moving into the instant
- the end of the take doesn't show the scene of the real photo
- text, captions or watermarks appear
Score how good a take it is overall (the moment, the action, the look). Reply with JSON only: {"pass": true or false,
"score": 1-10, "problem": "<the worst problem, one short sentence, or
empty>"}"""

QUIET = " A crisp camera lens shutter sound when the photo is taken. No speech, nobody talks."
# The soundtrack comes from MiniMax Music 3 (prompted by Gemini's "music") and is mixed under LTX-2.5's ambient
# sound by Momento Reveal. This is the fallback when the plan has no music.
MUSIC = ("Soft mellow lounge jazz, instrumental: warm Rhodes piano, upright bass, brushed drums, a gentle muted "
         "trumpet. Relaxed, nostalgic, unhurried, like a quiet afternoon.")
VIDEO_NEGATIVE = ("speech, talking, dialogue, voice, singing, vocals, music, pc game, console game, video game, cartoon, "
                  "childish, ugly")
