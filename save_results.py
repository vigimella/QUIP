import os, shutil
import matplotlib.pyplot as plt
import seaborn as sns


def create_file(execution_time, epochs, batch_size, learning_rate, threshold, model_train,
                model_test, output_file_training, confusion_matrix):
    with open(output_file_training, "w+") as file:
        file.write(f'Epochs: {epochs} \n')
        file.write(f'Batch Size: {batch_size} \n')
        file.write(f'Learning Rate: 0.{learning_rate} \n')
        file.write(f'Threshold: {threshold} \n')
        file.write('\n')
        file.write(f"train_loss: {model_train.history['loss']} \n")
        file.write(f"train_acc: {model_train.history['acc']} \n")
        file.write(f"train_prec: {model_train.history['prec']} \n")
        file.write(f"train_rec: {model_train.history['rec']} \n")
        file.write(f"val_loss: {model_train.history['val_loss']} \n")
        file.write(f"val_acc: {model_train.history['val_acc']} \n")
        file.write(f"val_prec: {model_train.history['val_prec']} \n")
        file.write(f"val_rec: {model_train.history['val_rec']} \n")
        file.write('\n')
        file.write(f'Test: {model_test} \n')
        file.write('\n')
        file.write(str(confusion_matrix))
        file.write('\n')
        file.write(f'Execution Time: {execution_time} \n')


def build_plot(folder, file_name, first_metric_list, second_metric_list, metric_name):
    plt.rcParams["figure.autolayout"] = True
    metric_name = metric_name.capitalize()
    file_name = file_name.replace('.results', '')
    plot_name = os.path.join(folder, f'{metric_name}_{file_name}')

    # plot creation
    num_epochs = len(first_metric_list) + 1
    epochs = range(1, num_epochs)

    fig, ax = plt.subplots()

    ax.plot(epochs, first_metric_list, color='#BF2A15', linestyle=':', label=f'Training {metric_name}')
    ax.plot(epochs, second_metric_list, 'o-', label=f'Validation {metric_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric_name}')
    ax.set_xticks(range(1, len(first_metric_list) + 1))
    ax.legend()

    # set plot colors

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#e5e5e5')
    ax.tick_params(color='#797979', labelcolor='#797979')
    ax.patch.set_edgecolor('black')
    ax.grid(True, color='white')

    if not num_epochs >= 20:
        plt.rcParams["figure.figsize"] = [(num_epochs / 3.22), 5.50]

    else:
        plt.tick_params(axis='x', which='major', labelsize=5)
        plt.rcParams["figure.figsize"] = [8.50, 5.50]

    plt.draw()

    fig.savefig(plot_name, dpi=180)

    print(f"{metric_name}'s plot created and stored at following path: {folder}.")


def build_confusion_matrix(confusion_matrix, class_labels, output_file_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_file_path)


def save_exp(files_folder, batch_size, epochs, learning_rate, timestamp, execution_time, threshold, model_train,
             model_test, confusion_matrix, class_labels):
    file_name = f'exp{str(batch_size)}{str(epochs)}{learning_rate}_{timestamp}_T{threshold}'

    new_folder = os.path.join(files_folder, file_name)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    output_file_resume = os.path.join(new_folder, f'experiment-{timestamp}.txt')
    output_confusion_matrix = os.path.join(new_folder, f'{timestamp}-confusion-matrix.png')
    create_file(execution_time, epochs, batch_size, learning_rate, threshold, model_train, model_test,
                output_file_resume, confusion_matrix)
    build_confusion_matrix(confusion_matrix, class_labels, output_confusion_matrix)

    build_plot(folder=new_folder, file_name=file_name, metric_name='accuracy',
               first_metric_list=model_train.history['acc'],
               second_metric_list=model_train.history['val_acc'])
    build_plot(folder=new_folder, file_name=file_name, metric_name='loss',
               first_metric_list=model_train.history['loss'],
               second_metric_list=model_train.history['val_loss'])
    
    return new_folder