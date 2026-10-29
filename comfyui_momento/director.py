"""Prompts, kept free of ComfyUI imports so scripts can read them."""

# What Gemini is told. It lives in the graph (Gemini's system prompt widget); this copy is the default.
DIRECTOR = """Left image: someone's profile picture. Right image: a photo they posted.
Describe the real moment that photo was taken, as a NEW camera angle on the same place: a bystander's view from
behind and to the side of the photographer, a few meters back. Everything the post shows (the subject, the buildings,
the objects, the light, the weather) is still there, now seen from this new position, along with the surroundings
the post cropped out. The photographer is the person in the profile picture (if it doesn't show a person, an
anthropomorphic rabbit in everyday clothes), holding a phone up, aimed at the subject of the post; the phone's screen
is visible and shows the shot being framed. Realistic, unhurried, flattering: it captures a nice moment. No jokes, no
mishaps, no dialogue. Reply with JSON only:
{"still": "<instruction for an image generator: one vertical frame of exactly that bystander view. Name every element
   of the post and where it sits in this new view, then the photographer and the phone. The person is <person_1> from
   ref_image_1 (same face, hair, build, clothes). This is a different camera position from ref_image_0, not a copy.>",
 "video": "<prompt for an image-to-video model, 4 seconds, no speech: the photographer steps into position, raises
   the phone, frames, and taps the shutter; the screen flashes. Handheld bystander camera. Sound: soft jazz playing
   quietly under the ambient sound of the place, and a crisp camera lens shutter sound at the moment the photo is
   taken. Nobody talks.>"}"""

QUIET = (" Soft mellow jazz plays quietly in the background. A crisp camera lens shutter sound when the photo is "
         "taken. No speech, nobody talks.")
VIDEO_NEGATIVE = ("speech, talking, dialogue, voice, singing, vocals, pc game, console game, video game, cartoon, "
                  "childish, ugly")
