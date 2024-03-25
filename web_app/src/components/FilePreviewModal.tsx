import { Modal, Paper, Text } from "@mantine/core";
import { MIME_TYPES } from "@mantine/dropzone";
import PDFViewer from "./PDFViewer.tsx";

interface FilePreviewModalProps {
  file: File;
  isModalOpen: boolean;
  setIsModalOpen: (isOpen: boolean) => void;
}

export const FilePreviewModal = ({
  file,
  isModalOpen,
  setIsModalOpen,
}: FilePreviewModalProps) => {
  const Preview = () => {
    if (file.type === MIME_TYPES.pdf) {
      return <PDFViewer file={file} />;
    }
    if (file.type === MIME_TYPES.mp4) {
      return <video controls src={URL.createObjectURL(file)} width="100%" />;
    }
    return <Text>File type not supported</Text>;
  };

  return (
    <Modal
      opened={isModalOpen}
      onClose={() => setIsModalOpen(false)}
      title={file.name}
      size="xl"
    >
      <Paper p="md">
        <Preview />
      </Paper>
    </Modal>
  );
};
