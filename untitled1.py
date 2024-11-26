test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")