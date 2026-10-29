"""Prompts, kept free of ComfyUI imports so scripts can read them."""

# What Gemini is told. It lives in the graph (Gemini's system prompt widget); this copy is the default.
DIRECTOR = """Left image: someone's profile picture. Right image: a photo they posted.
Describe the real moment that photo was taken, as a NEW camera angle on the same place: {angle} Everything the post
shows (the subject, the buildings, the objects, the light, the weather, the time of day) is still there, now seen from
this new position, along with the surroundings the post cropped out. The photographer is the person in the profile
picture (if it doesn't show a person, an anthropomorphic rabbit in everyday clothes), holding a phone up, aimed at the
subject of the post. Keep the person exactly as the profile picture shows them: same face, hair and build; don't add
jewelry, earrings, glasses, hats or accessories the profile picture doesn't show. Realistic, unhurried, flattering: it
captures a nice moment. No jokes, no mishaps, no dialogue. Reply with JSON only:
{{"still": "<instruction for an image generator: one vertical frame of exactly that view. Name every element of the
   post and where it sits in this new view, then the photographer and the phone. The person is <person_1> from
   ref_image_1. This is a different camera position from ref_image_0, not a copy.>",
 "video": "<prompt for an image-to-video model, 4 seconds, no speech: {motion} Sound: the ambient sound of the place,
   and a crisp camera lens shutter sound at the moment the photo is taken. Nobody talks.>",
 "music": "<a prompt for an instrumental music generator that fits THIS post's place, time and mood, from what you see:
   genre, instruments, tempo and feeling in one or two sentences (a punk show at night gets fast distorted punk; a
   quiet park at dawn gets something gentle; a city street gets the sound of that city).>"}}"""

# camera positions: the view Gemini plans, and what happens over the 4 seconds
ANGLES = {
    "behind": ("a bystander's view from behind and to the side of the photographer, a few meters back, with the phone's "
               "screen visible showing the shot being framed.",
               "the photographer steps into position, raises the phone and frames the shot while the handheld camera "
               "moves in over their shoulder, ending on exactly what the phone sees, the moment the shutter is "
               "tapped."),
    "face": ("a bystander's view from the side and slightly in front of the photographer, about two meters away, so "
             "their face is clearly visible in three-quarter view, concentrating, phone raised toward the subject, "
             "which is visible beside or behind them.",
             "the photographer raises the phone, frames the shot with a small concentrated look, taps the shutter and "
             "lowers the phone with a quiet smile. Handheld bystander camera, gently drifting."),
}


def director(angle="behind"):
    view, motion = ANGLES[angle]
    return DIRECTOR.format(angle=view, motion=motion)


QUIET = " A crisp camera lens shutter sound when the photo is taken. No speech, nobody talks."
# The soundtrack comes from MiniMax Music 3 (prompted by Gemini's "music") and is mixed under LTX-2.5's ambient
# sound by Momento Reveal. This is the fallback when the plan has no music.
MUSIC = ("Soft mellow lounge jazz, instrumental: warm Rhodes piano, upright bass, brushed drums, a gentle muted "
         "trumpet. Relaxed, nostalgic, unhurried, like a quiet afternoon.")
VIDEO_NEGATIVE = ("speech, talking, dialogue, voice, singing, vocals, music, pc game, console game, video game, cartoon, "
                  "childish, ugly")
