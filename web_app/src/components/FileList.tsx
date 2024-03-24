import cx from "clsx";
import { useState } from "react";
import {
  Table,
  ScrollArea,
  Modal,
  Paper,
  ActionIcon,
  Text,
  Group,
} from "@mantine/core";
import classes from "./FileList.module.css";
import { IconExternalLink, IconX } from "@tabler/icons-react";

export interface FileListProps {
  files: File[];
  setFiles: any;
}

export const FileList = ({ files, setFiles }: FileListProps) => {
  const [scrolled, setScrolled] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRemoveFile = (index: number) => {
    setFiles((prevFiles: File[]) => prevFiles.filter((_, i) => i !== index));
  };

  const handleFileClick = (file: File) => {
    setSelectedFile(file);
    setIsModalOpen(true);
  };

  const filesExpand = files.map((file, index) => (
    <Table.Tr key={file.name}>
      <Table.Td className={classes.tdName}>
        <Group>
          <Text>{file.name}</Text>
          <ActionIcon onClick={() => handleFileClick(file)} variant="light">
            <IconExternalLink />
          </ActionIcon>
        </Group>
      </Table.Td>
      <Table.Td>
        <ActionIcon
          color="red"
          variant="filled"
          onClick={() => handleRemoveFile(index)}
        >
          <IconX />
        </ActionIcon>
      </Table.Td>
    </Table.Tr>
  ));

  return (
    <>
      <ScrollArea
        onScrollPositionChange={({ y }) => setScrolled(y !== 0)}
        className={cx(classes.table)}
      >
        <Table>
          <Table.Thead
            className={`${cx(classes.header, { [classes.scrolled]: scrolled })}`}
          >
            <Table.Tr>
              <Table.Th className={classes.thName}>Name</Table.Th>
              <Table.Th className={classes.thRemove}>Remove</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>{filesExpand}</Table.Tbody>
        </Table>
      </ScrollArea>
      {selectedFile && (
        <Modal
          opened={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          title={selectedFile.name}
          size="xl"
        >
          <Paper p="md">
            <video
              controls
              src={URL.createObjectURL(selectedFile)}
              width="100%"
            />
          </Paper>
        </Modal>
      )}
    </>
  );
};
