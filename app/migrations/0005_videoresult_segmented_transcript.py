# Generated by Django 4.2.3 on 2023-08-15 05:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_videoresult'),
    ]

    operations = [
        migrations.AddField(
            model_name='videoresult',
            name='segmented_transcript',
            field=models.TextField(default=2),
            preserve_default=False,
        ),
    ]
