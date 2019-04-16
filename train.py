import os

import tensorflow as tf

import model
import a2c
import model_helper


def get_model_creator(flags):
    """Select the model class depending on flags."""
    if flags.model_name == 'policy_gradient':
        model_creator = model.PolicyGradient
    elif flags.model_name == 'a2c':
        model_creator = a2c.A2C
    else:
        raise ValueError("Unknown model name %s!" % flags.model_name)
        
    return model_creator

def train(flags):
    """Train the policy gradient model. """
    
    out_dir = flags.out_dir
    num_train_steps = flags.num_train_steps
    steps_per_infer = flags.steps_per_infer
    
    # Create model for train, infer mode
    model_creator = get_model_creator(flags)
    train_model = model_helper.create_train_model(flags, model_creator)
    infer_model = model_helper.create_infer_model(flags, model_creator)

    # TODO. set for distributed training and multi gpu 
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    
    # Session for train, infer
    train_sess = tf.Session(
        config=config_proto, graph=train_model.graph)
    infer_sess = tf.Session(
        config=config_proto, graph=infer_model.graph)
    
    # Load the train model if there's the file in the directory
    # otherwise, initialize vars in the train model
    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            out_dir, train_model.model, train_sess) 
    
    # Summary
    train_summary = "train_log"
    infer_summary = "infer_log"

    # Summary writer for train, infer
    train_summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, train_summary), train_model.graph)
    infer_summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, infer_summary))
    
    # First evaluation
    run_infer(infer_model, out_dir, infer_sess)

    # Initialize step var
    last_infer_steps = global_step
    
    # Training loop
    while global_step < num_train_steps:
        output_tuple = loaded_train_model.train(train_sess)
        global_step = output_tuple.global_step
        train_summary = output_tuple.train_summary    
    
        # Update train summary
        train_summary_writer.add_summary(train_summary, global_step)
        print('current global_step: {}'.format(global_step))

        # Evaluate the model for steps_per_infer 
        if global_step - last_infer_steps >= steps_per_infer:
            # Save checkpoint
            loaded_train_model.saver.save(train_sess, 
                os.path.join(out_dir, "rl.ckpt"), global_step)
            
            last_infer_steps = global_step
            output_tuple = run_infer(infer_model, out_dir, infer_sess)  
            infer_summary = output_tuple.infer_summary

            # Update infer summary
            infer_summary_writer.add_summary(infer_summary, global_step)
    
    # Done training
    loaded_train_model.saver.save(train_sess, 
        os.path.join(out_dir, "rl.ckpt"), global_step)
    print('Train done')
    
    
def run_infer(infer_model, model_dir, infer_sess):
    """Load vars. then, run infer model for 1 episode."""
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            model_dir, infer_model.model, infer_sess)
        
    output_tuple = loaded_infer_model.infer(infer_sess)
    return output_tuple
