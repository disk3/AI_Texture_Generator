import bpy


class ProgressDisplay:
    def __init__(self, context):
        self.wm = context.window_manager
        self._started = False

    def begin(self, steps: int = 100):
        self.wm.progress_begin(0, steps)
        self._started = True

    def update(self, value: int):
        if self._started:
            self.wm.progress_update(value)

    def end(self):
        if self._started:
            self.wm.progress_end()
            self._started = False

    def set_status(self, message: str):
        props = bpy.context.scene.ai_concept_props
        props.status_message = message

    def set_progress(self, progress: float):
        props = bpy.context.scene.ai_concept_props
        props.progress = progress
