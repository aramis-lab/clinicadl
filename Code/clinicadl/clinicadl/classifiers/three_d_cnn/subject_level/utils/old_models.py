from utils.modules import *


class Test2_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test2_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test2_batch_drop(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test2_batch_drop, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Dropout(p=0.5),
            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test2(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test31_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test31_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 4 * 5 * 4, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test31_batch_drop(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test31_batch_drop, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 4 * 5 * 4, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_0(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv5_FC3_0, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 4 * 5 * 4, 1000),
            nn.ReLU(),

            nn.Linear(1000, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC2_0(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv5_FC2_0, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 4 * 5 * 4, 500),
            nn.ReLU(),

            nn.Linear(500, 2)

        )

        self.flattened_shape = [-1, 64, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv6_FC2_0(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv6_FC2_0, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 1 * 2 * 1, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 64, 1, 2, 1]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv6_FC2_1(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv6_FC2_1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 3 * 4 * 3, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 3, 4, 3]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv6_FC2_2(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv6_FC2_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2)

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 1 * 2 * 1, 50),
            nn.LeakyReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 64, 1, 2, 1]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv7_FC2_0(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv7_FC2_0, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(64 * 2 * 2 * 2, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 2, 2, 2]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test3_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test3_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test3(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv4_FC4_0(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Conv4_FC4_0, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test4_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test4_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test4(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Optim(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Optim, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(16 * 41 * 51 * 44, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 16, 41, 51, 44]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Rieke(nn.Module):
    """
    Classifier for a multi-class classification task

    """
    def __init__(self, dropout=0.0, n_classes=2):
        super(Rieke, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(64 * 4 * 5 * 4, 128),
            nn.ReLU(),

            nn.Linear(128, n_classes)
        )

        self.flattened_shape = [-1, 64, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    def __len__(self):
        return len(self.layers)


class Test(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Test, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32)
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(32 * 4 * 5 * 4, 256),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(256, n_classes)
        )

        self.flattened_shape = [-1, 32, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test_nobatch(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Test_nobatch, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(32 * 5 * 6 * 5, 256),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(256, n_classes)
        )

        self.flattened_shape = [-1, 32, 5, 6, 5]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_3(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_3, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(32 * 20 * 25 * 21, 1000),
            nn.ReLU(),

            nn.Linear(1000, n_classes)
        )

        self.flattened_shape = [-1, 32, 20, 25, 21]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_4(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_4, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, n_classes)
        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_5(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_5, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(64 * 5 * 6 * 5, 1000),
            nn.ReLU(),

            nn.Linear(1000, n_classes)
        )

        self.flattened_shape = [-1, 4, 5, 4, 23]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Backstrom(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Backstrom, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 64, 7, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(128, 128, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(128, 128, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 11 * 13 * 12, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, n_classes)
        )

        self.flattened_shape = [-1, 128, 11, 13, 12]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Cheng(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Cheng, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 15, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(15, 25, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(25, 50, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(50, 50, 3, padding=1),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(50 * 11 * 13 * 12, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes)
        )

        self.flattened_shape = [-1, 50, 11, 13, 12]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x