
# The very first ones
# self.prompt_templates = ['a bad photo of a {}.',
#                          'a low resolution photo of the {}.',
#                          'a bad photo of the {}.',
#                          'a cropped photo of the {}.',
#                          ]
def generate_prompt_templates(prompt_type):
    if prompt_type == 'medical_low':
        prompt_normal = [
            'normal {}',
            'healthy {}',
            'intact {}',
            '{} with uniform structure',
            '{} showing clear tissue',
            '{} with normal anatomy',
            '{} showing no distortion',
            '{} with symmetric appearance',
            '{} with even texture',
        ]
        prompt_abnormal = [
            'abnormal {}',
            '{} with spot',
            '{} with abnormality',
            'diseased {}',
            '{} showing distortion',
            '{} with irregular area',
            '{} with irregular shape',
            '{} with uneven texture',
        ]
        # Prompt templates adapted for medical images
        prompt_templates = [
            'a medical image of a {}.',
            'a medical image of the {}.',
            'a diagnostic scan of a {}.',
            'a diagnostic scan of the {}.',
            'a slice showing a {}.',
            'a slice showing the {}.',
            'a scan of the {}.',
            'a clinical brain scan of a {}.'
        ]
    
    elif prompt_type == 'medical_high':
        prompt_normal = [
            '{} showing regular tissue appearance',
            '{} demonstrating standard anatomical appearance',
            'a {} image showing no pathological findings',
            'a {} scan demonstrating standard anatomical appearance',
            'a {} scan with intact brain structure and normal tissue contrast',
            'a clinically normal {} scan',
            'a {} image with physiologically typical anatomy'
        ]
        prompt_abnormal = [
            'an anomalous {} scan with tumor presence',
            'a {} image showing pathological intracranial findings',
            'a {} scan with neoplastic lesions',
            'a {} image indicating cerebral hemorrhage or tumor',
            'a {} scan with abnormal contrast indicative of disease',
            'a {} image presenting signs of brain pathology',
            'a {} scan showing mass effect or structural deviation',
            '{} with spot',
            '{} with lesion',
            '{} with tumor',
            '{} with hemorrhage',
            '{} demonstrating abnormal tissue mass',
        ]
        # Prompt templates adapted for medical images
        prompt_templates = [
            'a medical image of a {}.',
            'a medical image of the {}.',
            'a diagnostic scan of a {}.',
            'a diagnostic scan of the {}.',
            'a slice showing a {}.',
            'a slice showing the {}.',
            'a scan of the {}.',
            'a clinical brain scan of a {}.'
        ]
    
    elif prompt_type == 'object_agnostic':
        prompt_normal = ['{}']
        prompt_abnormal = ['damaged {}']
        prompt_templates = ['{}']
        
    elif prompt_type == 'industrial':
        prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                            '{} without defect',
                            '{} without damage']
        prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        prompt_templates = ['a photo of a {}.',
                                'a photo of the {}.',
                                'a cropped photo of a {}.',
                                'a cropped photo of the {}.',
                                'a close-up photo of a {}',
                                'a close-up photo of the {}',
                                'a photo of the {} for visual inspection',
                                'a photo of a {} for visual inspection',
                                ]
    return prompt_normal, prompt_abnormal, prompt_templates


    # # Normal vs abnormal prompts (medical style)
    # self.prompt_normal = [
    #     'normal {}',
    #     'healthy {}',
    #     '{} with uniform structure',
    #     '{} showing clear tissue',
    #     '{} with normal brain anatomy',
    #     '{} showing no distortion',
    #     '{} with symmetric appearance',
    #     '{} looking normal',
    # ]
    # self.prompt_abnormal = [
    #     'abnormal {}',
    #     '{} with lesion',
    #     '{} with tumor',
    #     '{} with hemorrhage',
    #     '{} with abnormality',
    #     'diseased {}',
    #     '{} showing distortion',
    #     '{} with strange shape',
    #     '{} with irregular area',
    #     '{} with structural abnormality',
    #     '{} showing diseased tissue',

    # ]
    # # Prompt templates adapted for medical images
    # self.prompt_templates = [
    #     'a medical image of a {}.',
    #     'a medical image of the {}.',
    #     'a diagnostic scan of a {}.',
    #     'a diagnostic scan of the {}.',
    #     'a slice showing a {}.',
    #     'a slice showing the {}.',
    #     'a CT scan of the {}.',
    #     'a clinical brain scan of a {}.'
    # ]

    ###################################################
    # self.prompt_normal = [
    #     'normal {}',
    #     'healthy {}',
    #     'clear {}',
    #     'clean {}',
    #     'smooth {}',
    #     '{} with sharp boundaries',
    #     '{} showing clear brain tissue',
    #     '{} with even texture',
    #     '{} with balanced gray levels',
    #     '{} with regular shape',
    #     '{} showing smooth patterns',
    #     '{} with normal brain structure',
    #     '{} scan with clean look',
    #     '{} with clear details',
    #     '{} looking uniform'
    # ]
    # self.prompt_abnormal = [
    #     'abnormal {}',
    #     '{} with spot',
    #     '{} with unclear boundary',
    #     '{} with mass',
    #     '{} with patchy area',
    #     '{} with irregular shape',
    #     '{} with distorted structure',
    #     '{} scan with strange region',
    #     '{} with uneven texture',
    #     '{} with extra tissue'
    # ]
    # self.prompt_templates = [
    #     'a scan of a {}.',
    #     'a scan of the {}.',
    #     'an image of a {}.',
    #     'an image of the {}.',
    #     'a slice of a {}.',
    #     'a slice of the {}.',
    #     'a brain scan showing a {}.',
    #     'a CT image of a {}.',
    #     'an MRI image of a {}.',
    #     'a cross-section of a {}.',
    #     'a close-up scan of a {}.',
    #     'a medical scan of a {}.',
    #     'a brain image of a {}.',
    #     'a diagnostic slice of a {}.'
    # ]