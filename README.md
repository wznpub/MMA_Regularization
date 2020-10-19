# MMA Regularization
The implementation for the NeurIPS2020 paper "MMA Regularization: Decorrelating Weights of Neural Networks by Maximizing the Minimal Angles", containing MMA regularization in PyTorch and in Symbol of MXNet.

# Usage
The usage of MMA Regularization in Pytorch, assuming applied to all layers:
        
        from MMA import get_mma_loss

        
        # in training method 
        ...
        ...     
           
        # normal learning loss
        loss = criterion(outputs, targets)
        
        # MMA Regularization
        for name, m in model.named_modules():
            # 'name' can be used to exclude some specified layers
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                mma_loss = get_mma_loss(m.weight)
                loss = loss + coefficient * mma_loss

        loss.backward()

The usage in other deep learning library, is similar. And the default coefficient is set to 0.07 for models without skip connections, and 0.03 for models with skip connections. However, it may need to be tuned according to different task, model, and dataset.
