import argparse
from multitask_prompting.trainer import Trainer

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm

from multitask_prompting.model import get_model
from multitask_prompting.data_utils import load_dataset, get_tokenized_dataloader

def main():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_dir', type=str, default='models')
    
    # experiment config
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--project_name', type=str, default='multitask_prompting')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=42)
    
    # args
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--dataset', type=str, default='nlu_evaluation_data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--model_type', type=str, default='fine_tuning')
    parser.add_argument('--base_plm_family', type=str, default='bert')
    parser.add_argument('--base_plm_path', type=str, default='bert-base-uncased')
    parser.add_argument("--tune_plm", type=bool, default=False)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--valid_split", type=float, default=0.3)
    parser.add_argument('--prompt_text', type=str, default='{"soft": None, "duplicate": 20}{"placeholder":"text_a"}{"mask"}.')
    parser.add_argument("--verbalizer_init", type=str, default ="{'alarm_query': 'alarm', 'alarm_remove': 'alarm', 'alarm_set': 'alarm', 'audio_volume_down': 'audio', 'audio_volume_mute': 'audio', 'audio_volume_other': 'audio', 'audio_volume_up': 'audio', 'calendar_query': 'calendar', 'calendar_remove': 'calendar', 'calendar_set': 'calendar', 'cooking_query': 'cooking', 'cooking_recipe': 'cooking', 'datetime_convert': 'datetime', 'datetime_query': 'datetime', 'email_addcontact': 'email', 'email_query': 'email', 'email_querycontact': 'email', 'email_sendemail': 'email', 'general_affirm': 'general', 'general_commandstop': 'general', 'general_confirm': 'general', 'general_dontcare': 'general', 'general_explain': 'general', 'general_greet': 'general', 'general_joke': 'general', 'general_negate': 'general', 'general_praise': 'general', 'general_quirky': 'general', 'general_repeat': 'general', 'iot_cleaning': 'iot', 'iot_coffee': 'iot', 'iot_hue_lightchange': 'iot', 'iot_hue_lightdim': 'iot', 'iot_hue_lightoff': 'iot', 'iot_hue_lighton': 'iot', 'iot_hue_lightup': 'iot', 'iot_wemo_off': 'iot', 'iot_wemo_on': 'iot', 'lists_createoradd': 'lists', 'lists_query': 'lists', 'lists_remove': 'lists', 'music_dislikeness': 'music', 'music_likeness': 'music', 'music_query': 'music', 'music_settings': 'music', 'news_query': 'news', 'play_audiobook': 'play', 'play_game': 'play', 'play_music': 'play', 'play_podcasts': 'play', 'play_radio': 'play', 'qa_currency': 'qa', 'qa_definition': 'qa', 'qa_factoid': 'qa', 'qa_maths': 'qa', 'qa_stock': 'qa', 'recommendation_events': 'recommendation', 'recommendation_locations': 'recommendation', 'recommendation_movies': 'recommendation', 'social_post': 'social', 'social_query': 'social', 'takeaway_order': 'takeaway', 'takeaway_query': 'takeaway', 'transport_query': 'transport', 'transport_taxi': 'transport', 'transport_ticket': 'transport', 'transport_traffic': 'transport', 'weather_query': 'weather'}")
    parser.add_argument("--max_seq_length", type=int, default=64)
    
    # hyperparams
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--prompt_learning_rate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)

    # optimizer 
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # steps
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()
    print(args.__dict__)

    set_seed(args.seed)

    plm, tokenizer, model_config, wrapper_class = load_plm(args.base_plm_family, args.base_plm_path)
    metadata, train_raw_dataset, eval_raw_dataset, test_raw_dataset = load_dataset(args)
    model = get_model(args.task, args.model)(args, plm, metadata, tokenizer, model_config, wrapper_class)
    train_dataloader, valid_dataloader, test_dataloader = get_tokenized_dataloader(args, train_raw_dataset, eval_raw_dataset, test_raw_dataset, model.tokenizer, model.template, model.wrapper_class)

    trainer = Trainer(args, model)
    if args.do_train:
        trainer.train(train_dataloader, valid_dataloader)
    
    if args.do_eval:
        trainer.test(args, model, test_dataloader)

