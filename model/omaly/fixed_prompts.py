def generate_prompt_templates(prompt_type):
    if prompt_type == 'medical':
        prompt_normal = [
            'normal {}',
            'intact {}',
            '{} with uniform structure',
            '{} showing clear tissue',
            '{} with normal anatomy',
            '{} showing no distortion',
            '{} with symmetric appearance',
            '{} looking normal',
            '{} with even texture',
            '{} with regular shape',
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

    elif prompt_type == 'object_agnostic':
        prompt_normal = ['{}']
        # prompt_normal = ['normal {}']
        prompt_abnormal = ['damaged {}']
        prompt_templates = ['{}']
        # prompt_templates = ['a photo of a {}']
        
    elif prompt_type == 'industrial':
        prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                            '{} without defect',
                            '{} without damage']
        prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        prompt_templates = ['a bad photo of a {}.',
                            'a low resolution photo of the {}.',
                            'a bad photo of the {}.',
                            'a cropped photo of the {}.',
                            ]        
    return prompt_normal, prompt_abnormal, prompt_templates
