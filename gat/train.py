import torch
from torch.utils.data import DataLoader
from data import make_subap_centers, SHWSGraphDataset3D, collate_graphs
from gshws import GSHWS_GAT_3D
from loss import WeightedMSELoss

torch.set_float32_matmul_precision('high')
def run_training_pipeline():
    # Target RTX 4050 directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Orchestrating pipeline execution context on: {device}")

    h5_dataset_path = "../Sum_NewData_299_5000.h5"
    focal_length = 20.0
    val_fraction = 0.2
    
    # 1. Coordinate and Data Mapping Setup
    subap_info = make_subap_centers(sub_num=12, pixel_lens=20)
    
    train_dataset = SHWSGraphDataset3D(h5_dataset_path, subap_info, focal_length, split='train', val_fraction=val_fraction)
    val_dataset = SHWSGraphDataset3D(h5_dataset_path, subap_info, focal_length, split='val', val_fraction=val_fraction)
    
    # Pre-calculate validation split frame delta count
    n_val_frames = int(5000 * val_fraction)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_graphs, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_graphs, num_workers=2, pin_memory=True)

    # 2. Compute Stabilized Statistical Weights from Data Store
    criterion = WeightedMSELoss.from_h5(h5_dataset_path, n_val=n_val_frames).to(device)

    # 3. Model Construction & Graph Compile Accelerator Optimization
    raw_model = GSHWS_GAT_3D(spot_feat_dim=3).to(device)
    
    # PyTorch 2.0 kernel compiler fusion for native speedups
    model = torch.compile(raw_model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    # 4. Training Real-time Execution Loop
    epochs = 10
    print("Initiating training loops...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            pred = model(
                batch['spot_feat'].to(device), 
                batch['subap_feat'].to(device), 
                batch['edge_index'].to(device), 
                batch['batch'].to(device)
            )
            
            loss = criterion(pred, batch['y'].to(device))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * (batch['batch'].max().item() + 1)

        # Validation Run Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch['spot_feat'].to(device), 
                    batch['subap_feat'].to(device), 
                    batch['edge_index'].to(device), 
                    batch['batch'].to(device)
                )
                loss = criterion(pred, batch['y'].to(device))
                total_val_loss += loss.item() * (batch['batch'].max().item() + 1)

        avg_train = total_train_loss / len(train_dataset)
        avg_val = total_val_loss / len(val_dataset)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:02d}/{epochs} | Stabilized Training Loss: {avg_train:.5f} | Validation Loss: {avg_val:.5f}")

if __name__ == '__main__':
    run_training_pipeline()