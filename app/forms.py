from django import forms
from .models import Video

class VideoForm(forms.ModelForm):
    # Choices for the VideoType dropdown
    VideoType_Choices = [
        ('Speech', 'Speech'),
        ('Webinar', 'Webinar'),
        ('Dialogue', 'Dialogue'),
        ('Podcast', 'Podcast'),
    ]

    VideoType = forms.ChoiceField(choices=VideoType_Choices, required=True)

    class Meta:
        model = Video
        fields = ['VideoTitle', 'VideoType', 'VideoURL', 'VideoTranscript']
